"""MTool dictionary translator using an OpenAI-compatible Chat Completions API."""
import argparse
import collections
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = '1.0'
DECISION_PROTOCOL = 'advisory-v1'
PROMPT_VERSION = 'dictionary-v1'
DEFAULT_API_BASE = 'https://api.deepseek.com'
DEFAULT_MODEL = 'deepseek-flash'
JP = re.compile(r'[\u3041-\u3096\u30a1-\u30fa\u3400-\u9fff]')
KANA = re.compile(r'[\u3041-\u3096\u30a1-\u30fa]')
PROTECTED = re.compile(r'\\(?:[A-Za-z]+(?:\[[^\]\r\n]*\]|<[^>\r\n]*>)?|[^\w\s])|\r\n|\n|\r|<[^>\r\n]+>|\{[^{}\r\n]+\}|%(?:\d+\$)?[sdif]|%\d+|\$\{[^}\r\n]+\}|@@P\d+@@')
SYSTEM = '''你是日语游戏的专业简体中文本地化译者。返回严格的 JSON 对象，不在 JSON 之外输出说明或 Markdown。
只翻译 items 中的文本，context_before/context_after 仅用于理解，不可输出。每条必须独立对应，不能合并、漏项、扩写或删减；自然地道，保持语气和含义。同一实体和同一词义尽量统一译法；术语表只是默认建议，可能切词错误或一词多义，不能为匹配字面而扭曲意思。不得用上下文替换当前文本或凭空补主语。
所有 @@P数字@@ 都是游戏控制符或变量的占位符，必须原样、原次数保留。仅 reorderable_placeholders 列出的编号变量可按中文语序调整位置；其余格式控制符必须保留原顺序。不翻译代码、资源路径和变量名。translations 的 ID 必须与本批 items 完全一致。
若不采用本条术语建议，在同一次响应中给出 term_exceptions，不要先强行使用错误译法。每个例外包括 term（日语词）、meaning（当前词义）、rendering（当前中文对应词，语法省略时可为空）、reason（为什么当前语境不适用建议）。例如 バー 可因“场所”译为酒吧、因“HP界面”译为条。专名一般保持一致，但若不是同一实体也应解释。只有例外需要解释，无冲突的条目不要写冗长理由。保留日文原样确有必要时，在 language_notes 中说明。
完整输出格式：{"translations":{"ID":"中文译文"},"term_exceptions":{"有例外的ID":[{"term":"日语词","meaning":"当前词义","rendering":"实际译法","reason":"具体语境理由"}]},"language_notes":{"必要时的ID":"保留日文的理由"}}。若收到语义复核请求，可反驳检测意见并保留原译；不要只为通过检测而换成建议词。若原译确实错误，才修正。
术语建议（日语→简体中文）：
'''


class TranslationError(Exception):
    pass


class Stopped(Exception):
    pass


class OutputError(TranslationError):
    pass


def read_json(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise TranslationError('JSON 含重复键，已停止，避免丢失数据。')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_credentials(api_base='', api_key='', model=''):
    base=((api_base or '').strip() or DEFAULT_API_BASE).rstrip('/')
    if base.endswith('/chat/completions'):
        base=base[:-len('/chat/completions')].rstrip('/')
    try:
        u=urllib.parse.urlsplit(base)
        valid=u.scheme in ('https','http') and bool(u.hostname) and not u.username and not u.password and not u.query and not u.fragment and u.port != 0
    except ValueError:
        valid=False
    if not valid:
        raise TranslationError('API Base 必须是有效的 HTTP(S) 地址，不能含账号、密码、查询参数或片段。')
    key=api_key.strip() if isinstance(api_key,str) else ''
    if not key:
        raise TranslationError('请填写 API Key。密钥仅在本次会话中使用，不会保存。')
    if '\r' in key or '\n' in key:
        raise TranslationError('API Key 不能包含换行。')
    selected=(model or '').strip() or DEFAULT_MODEL
    return key,base,selected


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TranslationError('API 出现重定向，已拒绝转发密钥。')


class API:
    def __init__(self, key, base, model, log=print, stop=None):
        self.key, self.base, self.model = key, base, model
        self.log, self.stop = log, stop or threading.Event()
        self.calls = 0
        self.usage = collections.Counter()
        self.opener = urllib.request.build_opener(NoRedirect())
        self.unsupported_parameters=set()
        self.completion_token_parameter='max_tokens'

    def check(self):
        if self.stop.is_set():
            raise Stopped()

    def complete(self, system, payload, max_tokens=8192):
        self.check()
        request_body={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'response_format':{'type':'json_object'},'temperature':0.2,'reasoning_effort':'none'}
        transient_failures=0
        for attempt in range(8):
            self.check()
            body_dict={k:v for k,v in request_body.items() if k not in self.unsupported_parameters}
            body_dict[self.completion_token_parameter]=max_tokens
            body=json.dumps(body_dict,ensure_ascii=False).encode('utf-8')
            req = urllib.request.Request(self.base + '/chat/completions', data=body, headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            try:
                with self.opener.open(req, timeout=120) as r:
                    result = json.loads(r.read())
                self.calls += 1
                for k, v in result.get('usage', {}).items():
                    if isinstance(v, (int, float)):
                        self.usage[k] += v
                choice = result['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise OutputError('API 输出被截断或未正常完成。')
                raw = choice['message'].get('content', '')
                try:
                    answer = json.loads(raw)
                except (ValueError, TypeError):
                    raise OutputError('API 返回的 JSON 无法解析。') from None
                if not isinstance(answer, dict):
                    raise OutputError('API 返回的 JSON 根节点不是对象。')
                return answer
            except urllib.error.HTTPError as e:
                status = e.code
                error_text=e.read(32768).decode('utf-8',errors='replace')
                e.close()
                if status in (400,422):
                    lower=error_text.lower()
                    unsupported=any(word in lower for word in ('not support','unsupported','unknown','unrecognized','not allowed','not permitted','only','extra','invalid'))
                    if unsupported and 'max_tokens' in lower and 'max_completion_tokens' in lower and self.completion_token_parameter=='max_tokens':
                        self.completion_token_parameter='max_completion_tokens'
                        self.log('接口要求 max_completion_tokens，已自动适配。')
                        continue
                    optional=next((name for name in ('response_format','temperature','reasoning_effort') if name in lower and name not in self.unsupported_parameters),None)
                    if unsupported and optional:
                        self.unsupported_parameters.add(optional)
                        self.log(f'接口不支持可选参数 {optional}，移除后重试。')
                        continue
                if status not in (408, 429, 500, 502, 503, 504):
                    meanings = {400: '请求参数或模型不受支持', 401: '密钥无效', 402: '余额不足', 403: '访问被拒绝', 404: '模型或接口不存在'}
                    raise TranslationError(f'API HTTP {status}：{meanings.get(status, "请求失败")}。') from None
                reason = f'HTTP {status}'
            except (urllib.error.URLError, TimeoutError, OSError):
                reason = '网络连接失败或超时'
            except (KeyError, IndexError, TypeError):
                raise TranslationError('API 响应结构异常。') from None
            transient_failures+=1
            if transient_failures>=4:
                raise TranslationError(reason + '，重试后仍失败。进度已保留。')
            delay = min(2 ** transient_failures, 20)
            self.log(f'{reason}，{delay} 秒后重试…')
            if self.stop.wait(delay):
                raise Stopped()
        raise TranslationError('接口兼容性重试次数已用尽，请检查 API Base 和模型能力。')


def protect(text):
    parts = []
    def sub(m):
        token = f'@@P{len(parts)}@@'
        parts.append(m.group())
        return token
    return PROTECTED.sub(sub, text), parts


def reorderable(part):
    return bool(re.fullmatch(r'%\d+(?:\$[sdif])?|\$\{[^}\r\n]+\}|\{[^{}\r\n]+\}', part))


def restore(text, parts):
    expected = [f'@@P{i}@@' for i in range(len(parts))]
    actual = re.findall(r'@@P\d+@@', text)
    if collections.Counter(actual) != collections.Counter(expected):
        raise TranslationError('控制符占位符发生丢失、重复或乱序。')
    fixed = {f'@@P{i}@@' for i, part in enumerate(parts) if not reorderable(part)}
    if [token for token in actual if token in fixed] != [token for token in expected if token in fixed]:
        raise TranslationError('格式控制符发生乱序。')
    return re.sub(r'@@P(\d+)@@', lambda m: parts[int(m.group(1))], text)


def matched_terms(source, terms):
    """Use longest, non-overlapping terms, so a compound overrides its component."""
    if not terms:
        return []
    pattern = '|'.join(re.escape(k) for k in sorted(terms, key=len, reverse=True))
    matches = []
    for m in re.finditer(pattern, source):
        term = m.group()
        if term != source:
            # A syllable inside a larger katakana word is not an independent term.
            if re.fullmatch(r'[ァ-ヺー]+', term):
                if (m.start() and re.match(r'[ァ-ヺー]', source[m.start()-1])) or (m.end() < len(source) and re.match(r'[ァ-ヺー]', source[m.end()])):
                    continue
            # Ordinary kana responses and single characters must not constrain unrelated words.
            if len(term) == 1 or re.fullmatch(r'[ぁ-ゖー]+', term):
                continue
        matches.append(term)
    return list(dict.fromkeys(matches))


def effective_glossary(terms):
    """Learn unambiguous compounds from existing paired labels, without changing the user's file."""
    candidates = collections.defaultdict(set)
    for source, target in terms.items():
        sm = re.fullmatch(r'(.+?)([0-9０-９]+)', source)
        tm = re.fullmatch(r'(.+?)([0-9０-９]+)', target)
        if sm and tm and sm[2] == tm[2]:
            candidates[sm[1]].add(tm[1])
        sp = re.findall(r'[（(]([^（）()]+)[）)]', source)
        tp = re.findall(r'[（(]([^（）()]+)[）)]', target)
        if len(sp) == len(tp):
            for a, b in zip(sp, tp):
                if JP.search(a) and not re.search(r'[\d_=<>]', a):
                    candidates[a].add(b)
    derived = {s: next(iter(values)) for s, values in candidates.items() if len(values) == 1 and s not in terms and 2 <= len(s) <= 24 and len(next(iter(values))) >= 2}
    return {**derived, **terms}


def validate_translation(source, target, terms):
    if not isinstance(target, str) or not target.strip():
        raise TranslationError('译文为空或类型错误。')
    source_codes, target_codes = PROTECTED.findall(source), PROTECTED.findall(target)
    if collections.Counter(source_codes) != collections.Counter(target_codes) or [p for p in source_codes if not reorderable(p)] != [p for p in target_codes if not reorderable(p)]:
        raise TranslationError('译文引入、丢失、改变或重排了控制符。')
    return target


def semantic_decision(source, target, terms, exceptions=None, language_note=''):
    """Strings identify questions; the model's contextual explanation resolves them."""
    accepted = []
    for entry in exceptions if isinstance(exceptions, list) else []:
        if not isinstance(entry, dict):
            continue
        term = entry.get('term')
        if not isinstance(term, str) or not term or term not in source:
            continue
        if not all(isinstance(entry.get(k), str) and entry[k].strip() for k in ('meaning', 'reason')):
            continue
        rendering = entry.get('rendering')
        if not isinstance(rendering, str):
            continue
        accepted.append({k:entry[k] for k in ('term','meaning','rendering','reason')})
    explained = {term for term in terms if any(term in entry['term'] for entry in accepted)}
    unresolved = []
    clean_source = PROTECTED.sub('', source)
    for term in matched_terms(clean_source, terms):
        if terms[term] not in target and term not in explained:
            unresolved.append({'kind':'term_suggestion', 'term':term, 'suggestion':terms[term], 'question':'当前语境是否为其他词义、复合词或自然表达？请保留正确译文并解释，或修正真正误译。'})
    note = language_note.strip() if isinstance(language_note,str) else ''
    remaining = PROTECTED.sub('',target)
    for entry in accepted:
        if entry['rendering']:
            remaining = remaining.replace(entry['rendering'],'')
    if KANA.search(remaining) and not note:
        unresolved.append({'kind':'remaining_japanese','question':'日文假名是否确有保留必要？若有，请说明；否则译成中文。'})
    return {'protocol':DECISION_PROTOCOL,'glossary_hash':digest(terms),'exceptions':accepted,'language_note':note,'unresolved':unresolved,'status':'needs_review' if unresolved else ('accepted_exception' if accepted else 'accepted')}


def remember_issue(api, ident, source, candidate, reason, attempts):
    if not hasattr(api, 'validation_issues'):
        api.validation_issues = {}
    api.validation_issues[source] = {'id': ident, 'source': source, 'candidate': candidate, 'reason': str(reason), 'attempts': attempts}


def clear_issue(api, source):
    getattr(api, 'validation_issues', {}).pop(source, None)


def format_semantic_log(source, translation, decision, ident=None):
    label=f'条目 {ident}' if ident is not None else '已保存记录'
    exceptions=decision.get('exceptions',[])
    title=f'{label}：已接受 {len(exceptions)} 项有语境说明的术语例外。' if exceptions else f'{label}：语义说明。'
    def block(label,value):
        return label+'\n    '+str(value).replace('\n','\n    ')
    lines=[title,block('原文：',source),block('AI 译文：',translation)]
    for number,entry in enumerate(exceptions,1):
        lines.extend([f'例外 {number} · 术语：{entry.get("term","")}',block('当前词义：',entry.get('meaning','')),block('本条译法：',entry.get('rendering','') or '随句意省略，不单独译出'),block('AI 理由：',entry.get('reason',''))])
    if decision.get('language_note'):
        lines.append(block('AI 保留原文的说明：',decision['language_note']))
    for issue in decision.get('unresolved',[]):
        lines.append(block('待确认的语义提示：',issue.get('question',str(issue))))
    return '\n'.join(lines)+'\n'+'─'*48


def saved_semantic_logs(output_path):
    out=Path(output_path)
    path=out.with_name(out.stem+'.work')/'semantic-decisions.json'
    data=read_json(path)
    records=data.get('records') if isinstance(data,dict) else None
    if not isinstance(records,list):
        raise TranslationError('已保存的语义记录格式不正确。')
    messages=[]
    count=0
    for number,entry in enumerate(records,1):
        if not isinstance(entry,dict) or not isinstance(entry.get('source'),str) or not isinstance(entry.get('translation'),str):
            raise TranslationError('已保存的语义记录缺少原文或译文。')
        messages.append(format_semantic_log(entry['source'],entry['translation'],entry,ident=f'历史 {number}'))
        count+=len(entry.get('exceptions',[]))
    return messages,count


def remember_semantic_context(api, source, decision):
    if decision and decision.get('exceptions'):
        if not hasattr(api,'semantic_memory'):
            api.semantic_memory={}
        api.semantic_memory[source]=decision['exceptions']


def relevant_semantic_context(api, group):
    current='\n'.join(source for _,source in group)
    examples=[]
    per_term=collections.Counter()
    for source,entries in reversed(list(getattr(api,'semantic_memory',{}).items())):
        selected=[]
        for entry in entries:
            if entry['term'] in current and per_term[entry['term']]<2:
                selected.append(entry)
                per_term[entry['term']]+=1
        if selected:
            examples.append({'source':source[:160],'contextual_exceptions':selected})
        if len(examples)>=6:
            break
    return examples


def is_resource(s):
    s = s.strip()
    return bool(re.fullmatch(r'[^\s]+\.(?:png|jpg|jpeg|webp|ogg|wav|mp3|m4a|js|json|rpgmvp|rpgmvo|ttf)', s, re.I) or re.match(r'^(?:img|audio|js|data|fonts)[/\\]', s) or re.match(r'^(?:MAIN|SUB|EV|MAP|SE|BGM)\d+[_-]', s, re.I))


def needs_translation(s):
    clean, _ = protect(s)
    clean = re.sub(r'@@P\d+@@', '', clean)
    return bool(JP.search(clean)) and not is_resource(s)


def load_dictionary(path):
    d = read_json(path)
    if not isinstance(d, dict) or any(not isinstance(v, str) for v in d.values()):
        raise TranslationError('需要“文本键 → 文本值”的 JSON 对象，所有值必须是字符串。')
    return d


def glossary_candidates(strings):
    counts = collections.Counter()
    for s in strings:
        counts.update(set(re.findall(r'[ァ-ヺー]{2,20}', s)))
    candidates = [k for k, n in counts.most_common(140) if n >= 2]
    # Short standalone labels often contain kanji names which katakana scanning misses.
    labels = [s for s in strings if 2 <= len(s) <= 10 and JP.search(s) and not is_resource(s) and not re.search(r'[。、！？\n<>_=/]', s)]
    return list(dict.fromkeys(candidates + labels[:80]))[:200]


def prepare_glossary(strings, path, api):
    if Path(path).exists():
        terms = read_json(path)
        if not isinstance(terms, dict) or any(not isinstance(v, str) or not k or not v.strip() for k, v in terms.items()):
            raise TranslationError('术语表应是非空字符串到非空字符串的 JSON 对象。')
        return terms
    candidates = glossary_candidates(strings)
    pool = [s for s in strings if needs_translation(s) and 12 <= len(s) <= 150]
    stride = max(1, len(pool) // 36)
    samples = pool[::stride][:36]
    api.log(f'建立全局术语表：分析 {len(candidates)} 个候选词和 {len(samples)} 条分散样本…')
    answer = api.complete('你是日语游戏简体中文本地化术语编辑。只返回 JSON：{"context":"不超过300字的作品背景、人物关系及翻译语气说明，只写样本有依据的信息，不要猜测", "glossary":{"日语术语":"统一中文译名"}}。结合候选词及分散样本确定人名、地名、种族、职业、物品、技能和固定界面术语；最多 90 项。仅收录输入里实际出现的短术语。不要收录完整句子、普通动词、拟声词、代码、文件名。译名应自然，同一人物和同一实体的默认译名保持一致。', {'candidates': candidates, 'samples': samples}, 8192)
    terms = answer.get('glossary')
    if not isinstance(terms, dict) or not terms:
        raise TranslationError('未能生成有效术语表，请重试或手工创建术语表。')
    corpus = '\n'.join(strings)
    terms = {k: v.strip() for k, v in terms.items() if isinstance(k, str) and isinstance(v, str) and 1 <= len(k) <= 24 and 1 <= len(v.strip()) <= 40 and k in corpus}
    if len(terms) > 90:
        ordered = sorted(terms, key=lambda k: (-sum(k in s for s in strings), len(k), k))
        terms = {k:terms[k] for k in ordered[:90]}
    if not terms:
        raise TranslationError('生成的术语表校验失败。')
    write_json(path, terms)
    context = answer.get('context', '')
    if isinstance(context, str) and context.strip():
        write_json(Path(path).with_name('context.json'), {'context': context[:1500]})
    api.log(f'已固定 {len(terms)} 个术语，可编辑：{path}')
    return terms


class Cache:
    def __init__(self, path, scope):
        self.db = sqlite3.connect(str(path))
        self.scope = scope
        self.meta_cache = {}
        self.db.execute('CREATE TABLE IF NOT EXISTS translations (scope TEXT, source TEXT, target TEXT, PRIMARY KEY(scope,source))')
        self.db.execute('CREATE TABLE IF NOT EXISTS decisions (scope TEXT, source TEXT, target_hash TEXT, payload TEXT, PRIMARY KEY(scope,source))')

    def get(self, s):
        row = self.db.execute('SELECT target FROM translations WHERE scope=? AND source=?', (self.scope, s)).fetchone()
        return row[0] if row else None

    def put(self, s, target, decision=None):
        self.db.execute('INSERT OR REPLACE INTO translations VALUES (?,?,?)', (self.scope, s, target))
        if decision is not None:
            self.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)',(self.scope,s,digest(target),json.dumps(decision,ensure_ascii=False)))
            self.meta_cache[s] = (digest(target),decision)
        self.db.commit()

    def decision(self, source, target):
        if source not in self.meta_cache:
            row=self.db.execute('SELECT target_hash,payload FROM decisions WHERE scope=? AND source=?',(self.scope,source)).fetchone()
            self.meta_cache[source] = (row[0],json.loads(row[1])) if row else None
        entry=self.meta_cache[source]
        return entry[1] if entry and entry[0]==digest(target) else None

    def close(self):
        self.db.close()


def translate_group(group, before, after, terms, cache, api, depth=0, repairs=None, semantic_review=False):
    items, parts = {}, {}
    for i, s in group:
        ident = str(i)
        items[ident], parts[ident] = protect(s)
    try:
        fixed = SYSTEM + json.dumps(terms, ensure_ascii=False, sort_keys=True) + '\n作品背景及语气（仅有依据的信息）：\n' + getattr(api, 'translation_context', '')
        payload = {'context_before': before, 'items': items, 'context_after': after}
        examples=relevant_semantic_context(api,group)
        if examples:
            payload['accepted_contextual_examples']=examples
            payload['examples_note']='这些是过去有理由接受的不同词义示例，仅作语境参考，不是全局替换规则。'
        movable = {ident:[f'@@P{i}@@' for i,part in enumerate(codes) if reorderable(part)] for ident,codes in parts.items() if any(reorderable(part) for part in codes)}
        if movable:
            payload['reorderable_placeholders'] = movable
        if repairs:
            payload['repair_instructions'] = ('这是语义疑点而非错误判决。请判断原译是否正确；若当前词义不适用术语建议，保留正确译文并返回 term_exceptions 的具体理由。不要为了字符串匹配强改译文。' if semantic_review else '上次输出有格式或控制符错误。请修复具体问题，并同时返回必要的术语例外理由。')
            payload['previous_validation_failures'] = repairs
        answer = api.complete(fixed, payload)
    except OutputError as e:
        if len(group) > 1:
            mid = len(group) // 2
            api.log('输出格式异常或超长，缩小批次重试。')
            return {**translate_group(group[:mid], before, after, terms, cache, api, depth+1), **translate_group(group[mid:], before, after, terms, cache, api, depth+1)}
        if depth < 2:
            return translate_group(group, before, after, terms, cache, api, depth+1)
        for i, s in group:
            remember_issue(api, i, s, '', e, depth+1)
        api.log(f'条目 {group[0][0]} 输出格式重试后仍失败，已记入待复查，继续后续条目。')
        return {}
    values = answer.get('translations')
    if not isinstance(values, dict) or set(values) != set(items):
        if len(group) > 1:
            mid = len(group) // 2
            api.log('本批条目对应不完整，拆成小批次重试。')
            return {**translate_group(group[:mid], before, after, terms, cache, api, depth+1), **translate_group(group[mid:], before, after, terms, cache, api, depth+1)}
        if depth < 2:
            return translate_group(group, before, after, terms, cache, api, depth+1, [{'id':group[0][0], 'reason':'必须原样返回 items 的 ID。'}])
        i, s = group[0]
        remember_issue(api, i, s, '', '单条翻译的 ID 校验失败。', depth+1)
        api.log(f'条目 {i} 的 ID 重试后仍不正确，已记入待复查，继续。')
        return {}
    good, bad, semantic_questions = {}, [], []
    for i, s in group:
        ident = str(i)
        try:
            t = values[ident]
            if not isinstance(t, str) or not t.strip():
                raise TranslationError('译文为空或类型错误。')
            t = restore(t, parts[ident])
            validate_translation(s, t, terms)
            exception_map=answer.get('term_exceptions',{})
            note_map=answer.get('language_notes',{})
            decision=semantic_decision(s,t,terms,exception_map.get(ident,[]) if isinstance(exception_map,dict) else [],note_map.get(ident,'') if isinstance(note_map,dict) else '')
            if decision['unresolved'] and not semantic_review:
                semantic_questions.append(((i,s),{'id':ident,'previous_translation':values[ident],'questions':decision['unresolved']}))
                continue
            good[s] = t
            cache.put(s, t, decision)
            remember_semantic_context(api,s,decision)
            if decision['exceptions'] or decision['unresolved'] or decision.get('language_note'):
                api.log(format_semantic_log(s,t,decision,ident=i))
            clear_issue(api, s)
        except TranslationError as e:
            candidate = values.get(ident)
            remember_issue(api, i, s, candidate, e, depth+1)
            bad.append(((i, s), {'id':ident, 'previous_translation':candidate, 'reason':str(e), 'required_terms':{k:terms[k] for k in matched_terms(s, terms)}}))
            api.log(f'条目 {i} 未通过校验：{e}')
    if bad:
        if depth >= 2:
            api.log(f'{len(bad)} 条重试后仍未通过，已记入待复查，继续后续条目。')
        else:
            api.log(f'{len(bad)} 条格式未通过校验，带具体失败原因重试…')
            for pair, repair in bad:
                good.update(translate_group([pair], before, after, terms, cache, api, depth+1, [repair], semantic_review))
    if semantic_questions:
        api.log(f'{len(semantic_questions)} 条有语义疑点，合并询问一次；允许说明理由后保留原译。')
        good.update(translate_group([pair for pair,_ in semantic_questions],before,after,terms,cache,api,depth,[repair for _,repair in semantic_questions],True))
    return good


@contextlib.contextmanager
def job_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    locked = False
    try:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                raise TranslationError('另一个程序正在处理相同输出，请等待它结束。') from None
        yield
    finally:
        if locked:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def run_translation(input_path, output_path, api, batch_size=24, char_limit=4000, limit=0, glossary_only=False, accept_partial=False):
    out = Path(output_path).resolve()
    if out == Path(input_path).resolve():
        raise TranslationError('输出文件不能与输入文件相同，原文件必须保留。')
    with job_lock(out.with_name(out.stem + '.work') / 'run.lock'):
        return _run_translation(input_path, output_path, api, batch_size, char_limit, limit, glossary_only, accept_partial)


def _run_translation(input_path, output_path, api, batch_size=24, char_limit=4000, limit=0, glossary_only=False, accept_partial=False):
    inp, out = Path(input_path).resolve(), Path(output_path).resolve()
    if inp == out:
        raise TranslationError('输出文件不能与输入文件相同，原文件必须保留。')
    if not 1 <= batch_size <= 100 or not 200 <= char_limit <= 16000 or limit < 0:
        raise TranslationError('批量大小、字符上限或测试条数无效。')
    d = load_dictionary(inp)
    original_bytes = inp.read_bytes()
    work = out.with_name(out.stem + '.work')
    work.mkdir(parents=True, exist_ok=True)
    glossary_path = work / 'glossary.json'
    strings = list(dict.fromkeys(v for v in d.values() if needs_translation(v)))
    api.log(f'共 {len(d)} 个键；需翻译的不同文本 {len(strings)} 条。数值、资源路径及纯符号保留。')
    terms = prepare_glossary(strings, glossary_path, api) if strings else {}
    if glossary_only:
        api.log('术语表已就绪。可先编辑再开始正式翻译。')
        return {'glossary': str(glossary_path), 'terms': len(terms)}
    context_path = work / 'context.json'
    context = read_json(context_path).get('context', '') if context_path.exists() else ''
    if not isinstance(context, str):
        raise TranslationError('context.json 的 context 必须是字符串。')
    api.translation_context = context
    scope = digest([PROMPT_VERSION, api.base, api.model, terms, context])
    effective_terms = effective_glossary(terms)
    overrides_path = work / 'glossary-overrides.json'
    if overrides_path.exists():
        overrides = read_json(overrides_path)
        if not isinstance(overrides, dict) or any(not isinstance(v,str) or not k or not v.strip() for k,v in overrides.items()):
            raise TranslationError('glossary-overrides.json 应是日文到中文的字符串映射。')
        effective_terms.update(overrides)
        api.log(f'已加载 {len(overrides)} 项术语建议修正；词义分歧允许解释后保留。')
    if len(effective_terms) > len(terms):
        api.log(f'已从现有术语标签推导 {len(effective_terms)-len(terms)} 个无歧义复合词，避免拆词误判。')
    cache = Cache(work / 'cache.sqlite3', scope)
    pending_path = out.with_name(out.stem + '.partial.json')
    report_path = work / 'report.json'
    issues_path = work / 'issues.json'
    decisions_path = work / 'semantic-decisions.json'
    api.validation_issues = {}
    api.semantic_memory = {}
    if issues_path.exists():
        saved = read_json(issues_path)
        if isinstance(saved, dict):
            api.validation_issues = {entry['source']:entry for entry in saved.get('pending', []) if isinstance(entry, dict) and entry.get('source') in strings}
    result = dict(d)
    translated = {}
    attempted = 0
    started = time.time()
    completed = False
    def compatible_cached(source):
        target = cache.get(source)
        if target is not None:
            try:
                validate_translation(source, target, effective_terms)
                return target
            except TranslationError:
                pass
        return None
    def export():
        for k, source in d.items():
            result[k] = translated.get(source, source)
        if list(result) != list(d) or any(not isinstance(v, str) for v in result.values()):
            raise TranslationError('最终键集合或值类型校验失败。')
        write_json(pending_path, result)
        pending = [entry for source, entry in api.validation_issues.items() if source not in translated]
        write_json(issues_path, {'pending_count':len(pending), 'pending':pending})
        semantic_records=[]
        reviewed=0
        for source,target in translated.items():
            decision=cache.decision(source,target)
            if decision and decision.get('protocol')==DECISION_PROTOCOL:
                reviewed+=1
                if decision.get('exceptions') or decision.get('unresolved'):
                    semantic_records.append({'source':source,'translation':target,**decision})
        write_json(decisions_path,{'records':semantic_records})
        report = {'version': VERSION, 'input': str(inp), 'input_sha256': hashlib.sha256(original_bytes).hexdigest(), 'output': str(out), 'complete': completed, 'keys': len(d), 'unique_translatable': len(strings), 'translated_unique': len(translated), 'remaining_unique': len(strings)-len(translated), 'validation_issue_count':len(pending), 'issues_file':str(issues_path), 'api_calls_this_run': api.calls, 'usage_this_run': dict(api.usage), 'elapsed_seconds': round(time.time()-started, 1), 'model': api.model, 'glossary_sha256': digest(terms), 'effective_glossary_sha256': digest(effective_terms)}
        report.update({'semantic_reviewed_unique':reviewed,'semantic_exception_count':sum(len(entry.get('exceptions',[])) for entry in semantic_records),'semantic_warning_count':sum(bool(entry.get('unresolved')) for entry in semantic_records),'semantic_decisions_file':str(decisions_path)})
        write_json(report_path, report)
        return report
    try:
        for s in strings:
            cached = compatible_cached(s)
            if cached is not None:
                translated[s] = cached
                remember_semantic_context(api,s,cache.decision(s,cached))
                clear_issue(api, s)
        todo = [(i, s) for i, s in enumerate(strings) if s not in translated]
        api.log(f'已复用 {len(translated)} 条缓存/术语，待请求 {len(todo)} 条。')
        pos = 0
        while pos < len(todo):
            api.check()
            if limit and attempted >= limit:
                break
            group, chars = [], 0
            while pos < len(todo) and len(group) < batch_size:
                pair = todo[pos]
                if group and chars + len(pair[1]) > char_limit:
                    break
                if limit and attempted + len(group) >= limit:
                    break
                group.append(pair)
                chars += len(pair[1])
                pos += 1
            first, last = group[0][0], group[-1][0]
            before = [{'source': s, 'translation': translated.get(s, '')} for s in strings[max(0, first-3):first]]
            after = strings[last+1:last+4]
            translated.update(translate_group(group, before, after, effective_terms, cache, api))
            attempted += len(group)
            export()
            api.log(f'进度 {len(translated)}/{len(strings)}；待复查 {len(api.validation_issues)} 条；本次 API 请求 {api.calls} 次。')
        completed = len(translated) == len(strings)
        report = export()
        if report.get('semantic_warning_count'):
            api.log(f'{report["semantic_warning_count"]} 条语义提示已保存到 {decisions_path}；没有按词面建议强制改写。')
        if inp.read_bytes() != original_bytes:
            completed = False
            raise TranslationError('运行中输入文件被修改，输出只保留为 partial，请重新运行。')
        if completed or accept_partial:
            os.replace(pending_path, out)
            api.log(('翻译完成：' if completed else '测试样本已导出（包含未译原文）：') + str(out))
        else:
            if api.validation_issues:
                api.log(f'本轮结束，{len(api.validation_issues)} 条待复查，详情：{issues_path}。已保存 partial；再次开始可重试未完成条目。')
            else:
                api.log('样本翻译已保存为 partial 文件；正式输出需全部翻译完成。')
        return report
    finally:
        # Also collect successful individual retries when a later item fails or the user stops.
        for s in strings:
            if s not in translated:
                t = compatible_cached(s)
                if t is not None:
                    translated[s] = t
        if not completed:
            export()
        cache.close()


def self_check():
    import tkinter
    return {'version': VERSION, 'python': sys.version.split()[0], 'sqlite': sqlite3.sqlite_version, 'tcl': tkinter.Tcl().eval('info patchlevel'), 'frozen': bool(getattr(sys, 'frozen', False))}


def gui(hidden_test_path=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    if hidden_test_path:
        root.withdraw()
    root.title('MTool 文本翻译器 v' + VERSION)
    root.geometry('880x760')
    root.minsize(760, 680)
    root.option_add('*Font', ('Microsoft YaHei UI', 10))
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='日语游戏文本 → 简体中文', font=('Microsoft YaHei UI', 17, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='保持所有键名原样 · 术语建议与语境例外 · 分批上下文 · 缓存与断点续译').pack(anchor='w', pady=(5, 15))
    fields = {}
    class PlaceholderEntry(tk.Entry):
        def __init__(self,parent,hint,secret=False):
            super().__init__(parent,font=('Microsoft YaHei UI',10),relief='solid',borderwidth=1)
            self.hint,self.secret,self.showing_hint=hint,secret,False
            self.bind('<FocusIn>',self.hide_hint)
            self.bind('<FocusOut>',self.show_hint)
            self.show_hint()
        def get(self):
            return '' if self.showing_hint else super().get()
        def hide_hint(self,event=None):
            if self.showing_hint:
                self.delete(0,'end')
                self.showing_hint=False
                self.configure(fg='#111111',show='●' if self.secret else '')
        def show_hint(self,event=None):
            if not super().get():
                self.showing_hint=True
                self.configure(fg='#888888',show='')
                self.insert(0,self.hint)
    def api_field(label,hint,secret=False):
        row=ttk.Frame(frame)
        row.pack(fill='x',pady=4)
        ttk.Label(row,text=label,width=13).pack(side='left')
        entry=PlaceholderEntry(row,hint,secret)
        entry.pack(side='left',fill='x',expand=True,ipady=4)
        return entry
    def field(label, value, chooser=None, secret=False):
        row = ttk.Frame(frame)
        row.pack(fill='x', pady=4)
        ttk.Label(row, text=label, width=13).pack(side='left')
        var = tk.StringVar(value=value)
        entry = ttk.Entry(row, textvariable=var, show='●' if secret else '')
        entry.pack(side='left', fill='x', expand=True)
        if chooser:
            ttk.Button(row, text='选择…', command=lambda: chooser(var)).pack(side='left', padx=(8, 0))
        fields[label] = var
        return var
    def choose_input(var):
        p = filedialog.askopenfilename(filetypes=[('JSON 文件', '*.json')])
        if p:
            var.set(p)
            output.set(str(Path(p).with_name(Path(p).stem + '.zh-CN.json')))
    inp = field('输入 JSON', '', choose_input)
    output = field('输出 JSON', '', lambda v: v.set(filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON 文件','*.json')]) or v.get()))
    api_base=api_field('API Base',DEFAULT_API_BASE)
    key=api_field('API Key','必填 · 仅本次会话使用，不保存',True)
    model=api_field('Model Name',DEFAULT_MODEL)
    ttk.Label(frame,text='Base 和模型留空时使用灰色提示的默认值；支持 OpenAI 兼容的 Chat Completions 接口。',wraplength=780).pack(anchor='w',pady=5)
    options = ttk.Frame(frame)
    options.pack(fill='x', pady=7)
    ttk.Label(options, text='每批条数').pack(side='left')
    batch = tk.StringVar(value='24')
    ttk.Spinbox(options, from_=1, to=100, textvariable=batch, width=5).pack(side='left', padx=(8, 18))
    ttk.Label(options, text='仅测试条数（0=全部）').pack(side='left')
    limit = tk.StringVar(value='0')
    ttk.Spinbox(options, from_=0, to=100000, textvariable=limit, width=7).pack(side='left', padx=8)
    ttk.Label(frame, text='术语表提供默认建议；不同词义可由 AI 说明理由后保留，理由随缓存保存。停止后再次开始会自动续译。', wraplength=790).pack(anchor='w', pady=5)
    buttons = ttk.Frame(frame)
    buttons.pack(fill='x', pady=8)
    history_buttons=ttk.Frame(frame)
    history_buttons.pack(fill='x',pady=(0,4))
    events = queue.Queue()
    stop = threading.Event()
    active = [False]
    logframe=ttk.Frame(frame)
    logframe.pack(fill='both',expand=True,pady=8)
    scrollbar=ttk.Scrollbar(logframe,orient='vertical')
    scrollbar.pack(side='right',fill='y')
    logbox = tk.Text(logframe, height=14, wrap='word', state='disabled', font=('Microsoft YaHei UI', 10),yscrollcommand=scrollbar.set)
    logbox.pack(side='left',fill='both', expand=True)
    scrollbar.configure(command=logbox.yview)
    status = tk.StringVar(value='就绪。原文件始终保留；全部成功后才生成正式输出。')
    ttk.Label(frame, textvariable=status, wraplength=780).pack(anchor='w')
    def log(s):
        events.put(('log', s))
    def start(only=False):
        if active[0]:
            return
        try:
            credentials = load_credentials(api_base.get(), key.get(), model.get())
            ip, op = inp.get(), output.get()
            bs, lim = int(batch.get()), int(limit.get())
            if not Path(ip).is_file():
                raise TranslationError('请选择存在的输入 JSON。')
            if Path(ip).resolve() == Path(op).resolve():
                raise TranslationError('输出不能覆盖原文件。')
            if Path(op).exists() and not only and not messagebox.askyesno('覆盖输出文件', '已有同名输出文件，翻译完成后替换它？原始输入文件不受影响。'):
                return
        except (ValueError, OSError, TranslationError) as e:
            messagebox.showerror('无法开始', str(e))
            return
        active[0] = True
        stop.clear()
        for b in run_buttons:
            b.configure(state='disabled')
        status.set('正在后台处理；停止会在当前请求结束后保存进度。')
        def worker():
            try:
                api = API(*credentials, log=log, stop=stop)
                report = run_translation(ip, op, api, batch_size=bs, limit=lim, glossary_only=only)
                if only:
                    message = '术语表已生成。'
                elif report['complete']:
                    message = f'翻译完成；另有 {report["semantic_warning_count"]} 条语义提示，见 semantic-decisions.json。' if report.get('semantic_warning_count') else '翻译完成。术语例外及理由已随进度保存。'
                elif report.get('validation_issue_count'):
                    message = f'本轮结束，仍有 {report["validation_issue_count"]} 条待复查，详情见进度目录的 issues.json；再次开始可重试。'
                else:
                    message = '测试完成，结果见 .partial.json，正式翻译请将测试条数改为 0。'
                events.put(('done', message))
            except Stopped:
                events.put(('done', '已停止，进度已保存。再次开始即可继续。'))
            except Exception as e:
                events.put(('error', safe_error(e, credentials[0])))
        threading.Thread(target=worker, daemon=True).start()
    run_buttons = [ttk.Button(buttons, text='① 生成 / 检查术语表', command=lambda: start(True)), ttk.Button(buttons, text='② 开始 / 继续翻译', command=start)]
    for b in run_buttons:
        b.pack(side='left', padx=(0, 8))
    ttk.Button(buttons, text='停止并保存', command=stop.set).pack(side='left')
    def open_work():
        if not output.get():
            log('请先选择输入文件，或填写对应的输出 JSON 路径。')
            return
        p = Path(output.get()).with_name(Path(output.get()).stem + '.work')
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))
    def show_saved_history():
        try:
            if not output.get():
                raise TranslationError('请先选择输入文件，或填写已完成翻译的输出 JSON 路径。')
            messages,count=saved_semantic_logs(output.get())
            log(f'读取已保存的 {len(messages)} 条语义记录，共 {count} 项术语例外；仅在本地查看，不调用 API。')
            for message in messages:
                log(message)
            if not active[0]:
                status.set(f'已显示 {len(messages)} 条已保存记录，共 {count} 项术语例外；未进行翻译。')
            return len(messages),count
        except (OSError,ValueError,TranslationError) as e:
            log('无法读取已保存的语义记录：'+str(e))
            return 0,0
    history_button=ttk.Button(history_buttons,text='查看已保存的术语例外（原文 / 译文 / 理由）',command=show_saved_history)
    history_button.pack(side='left')
    ttk.Button(history_buttons, text='打开术语及进度目录', command=open_work).pack(side='right')
    def poll():
        while not events.empty():
            kind, msg = events.get()
            logbox.configure(state='normal')
            logbox.insert('end', time.strftime('%H:%M:%S ') + msg + '\n')
            logbox.see('end')
            logbox.configure(state='disabled')
            if kind in ('done', 'error'):
                active[0] = False
                for b in run_buttons:
                    b.configure(state='normal')
                status.set(msg)
        root.after(150, poll)
    def close():
        if active[0]:
            stop.set()
            status.set('正在停止并保存。请等当前请求结束后关闭窗口。')
        else:
            root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    if hidden_test_path:
        history_count=exception_count=0
        saved=Path(output.get()).with_name(Path(output.get()).stem+'.work')/'semantic-decisions.json' if output.get() else None
        if saved and saved.exists():
            history_count,exception_count=show_saved_history()
            poll()
        root.update_idletasks()
        write_json(hidden_test_path, {'version':VERSION,'gui_constructed': True, 'run_buttons': len(run_buttons), 'default_input': inp.get(), 'default_output': output.get(),'api_base_is_empty':not api_base.get(),'model_is_empty':not model.get(),'api_key_is_empty':not key.get(),'api_base_placeholder':api_base.hint,'model_placeholder':model.hint,'placeholders_are_gray':api_base.cget('fg')=='#888888' and model.cget('fg')=='#888888','window_hidden': root.state() == 'withdrawn','history_records':history_count,'history_exceptions':exception_count,'history_displays_original_translation_reason':all(label in logbox.get('1.0','end') for label in ('原文：','AI 译文：','AI 理由：')) if exception_count else True})
        root.destroy()
        return
    poll()
    root.mainloop()


def safe_error(e, key=''):
    msg = str(e)
    if key:
        msg = msg.replace(key, '[密钥已隐藏]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[密钥已隐藏]', msg)


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description='MTool JSON 文本翻译器：保持键名不变，使用可配置的兼容接口翻译文本值')
    p.add_argument('--input')
    p.add_argument('--output')
    p.add_argument('--api-base',default='',help='兼容接口的 Base URL，留空使用内置默认值')
    p.add_argument('--api-key-env',default='MTOOL_TRANSLATOR_API_KEY',help='命令行模式从此环境变量读取密钥')
    p.add_argument('--model', default='')
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--char-limit', type=int, default=4000)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--glossary-only', action='store_true')
    p.add_argument('--accept-partial', action='store_true', help='仅测试用途，允许导出含未译原文的正式文件')
    p.add_argument('--self-check', metavar='REPORT_JSON')
    p.add_argument('--gui-self-check', metavar='REPORT_JSON', help=argparse.SUPPRESS)
    p.add_argument('--log', help='后台命令行运行时写入日志')
    args = p.parse_args()
    if args.self_check:
        write_json(args.self_check, self_check())
        return 0
    if args.gui_self_check:
        gui(args.gui_self_check)
        return 0
    if not args.input:
        gui()
        return 0
    logfile = open(args.log, 'a', encoding='utf-8') if args.log else None
    def log(s):
        if sys.stdout:
            print(s, flush=True)
        if logfile:
            logfile.write(time.strftime('%Y-%m-%d %H:%M:%S ') + s + '\n')
            logfile.flush()
    key = ''
    try:
        key, base, model = load_credentials(args.api_base,os.environ.get(args.api_key_env,''),args.model)
        out = args.output or str(Path(args.input).with_name(Path(args.input).stem + '.zh-CN.json'))
        run_translation(args.input, out, API(key, base, model, log), args.batch_size, args.char_limit, args.limit, args.glossary_only, args.accept_partial)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        log('已停止：' + safe_error(e, key))
        return 1
    finally:
        if logfile:
            logfile.close()


if __name__ == '__main__':
    raise SystemExit(main())
