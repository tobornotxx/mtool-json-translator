import json
import io
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch
import translator as t

test_temp = Path(__file__).parent / 'test_temp'
test_temp.mkdir(exist_ok=True)
tempfile.tempdir = str(test_temp)


class FakeAPI:
    base = 'https://api.example.test/v1'
    model = 'test'
    calls = 0
    usage = {}
    def __init__(self):
        self.stop = threading.Event()
        self.requests = []
        self.validation_issues = {}
        self.calls = 0
        self.usage = {}
    def log(self, s):
        pass
    def check(self):
        if self.stop.is_set():
            raise t.Stopped()
    def complete(self, system, payload, *args):
        self.check()
        self.calls += 1
        self.requests.append(payload)
        return {'translations': {k: v.replace('ルナ', '露娜').replace('村へ行く', '前往村庄').replace('休む', '休息') for k, v in payload['items'].items()}}


class Tests(unittest.TestCase):
    def test_roundtrip_codes_and_markers(self):
        text = r'\C[3]ルナ\V[1] %1 {name} <b>' + '\r\n' + '@@P0@@'
        masked, parts = t.protect(text)
        self.assertEqual(t.restore(masked, parts), text)
        with self.assertRaises(t.TranslationError):
            t.restore(masked + '@@P0@@', parts)

    def test_no_duplicates_no_nested_or_nonstring(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'in.json'
            for raw in ['{"a":"x","a":"y"}', '{"x":1}', '["x"]']:
                p.write_text(raw, encoding='utf-8')
                with self.assertRaises(t.TranslationError):
                    t.load_dictionary(p)

    def setup_job(self, root):
        inp, out = root/'in.json', root/'out.json'
        d = {'ルナ': 'ルナ', '村へ行く': '村へ行く', 'same-source': '村へ行く', '休む': '休む', '42': '42', r'\C[1]休む\C[0]': r'\C[1]休む\C[0]', 'already-translated': '已经翻译'}
        t.write_json(inp, d)
        work = root/'out.work'
        work.mkdir()
        t.write_json(work/'glossary.json', {'ルナ': '露娜'})
        return inp, out, d

    def test_partial_resume_duplicates_key_order_and_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, out, d = self.setup_job(root)
            raw = inp.read_bytes()
            a = FakeAPI()
            r = t.run_translation(inp, out, a, limit=1)
            self.assertFalse(r['complete'])
            self.assertFalse(out.exists())
            self.assertTrue((root/'out.partial.json').exists())
            r = t.run_translation(inp, out, a)
            self.assertTrue(r['complete'])
            actual = t.read_json(out)
            self.assertEqual(list(actual), list(d))
            self.assertEqual(actual['村へ行く'], actual['same-source'])
            self.assertEqual(actual['42'], '42')
            self.assertEqual(actual[r'\C[1]休む\C[0]'], r'\C[1]休息\C[0]')
            self.assertEqual(actual['already-translated'], '已经翻译')
            self.assertEqual(inp.read_bytes(), raw)
            a2 = FakeAPI()
            t.run_translation(inp, out, a2)
            self.assertEqual(a2.calls, 0)

    def test_changed_glossary_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, out, _ = self.setup_job(root)
            t.run_translation(inp, out, FakeAPI())
            t.write_json(root/'out.work'/'glossary.json', {'ルナ':'露娜', '別名':'别名'})
            a = FakeAPI()
            t.run_translation(inp, out, a)
            self.assertGreater(a.calls, 0)

    def test_stop_preserves_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, out, _ = self.setup_job(root)
            a = FakeAPI()
            a.stop.set()
            with self.assertRaises(t.Stopped):
                t.run_translation(inp, out, a)
            self.assertTrue((root/'out.partial.json').exists())

    def test_bad_output_never_cached(self):
        with tempfile.TemporaryDirectory() as td:
            cache = t.Cache(Path(td)/'cache.db', 'test')
            a = FakeAPI()
            a.complete = lambda *args, **kw: {'translations':{'1':'坏@@P5@@'}}
            self.assertEqual(t.translate_group([(1,'休む')], [], [], {}, cache, a), {})
            self.assertIsNone(cache.get('休む'))
            self.assertIn('休む', a.validation_issues)
            cache.close()

    def test_refuse_input_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'in.json'
            t.write_json(p, {'ルナ':'ルナ'})
            with self.assertRaises(t.TranslationError):
                t.run_translation(p, p, FakeAPI())

    def test_provider_configuration_and_empty_defaults(self):
        self.assertEqual(t.load_credentials('','test-key',''),('test-key',t.DEFAULT_API_BASE,t.DEFAULT_MODEL))
        self.assertEqual(t.load_credentials('  ',' test-key ','  '),('test-key',t.DEFAULT_API_BASE,t.DEFAULT_MODEL))
        self.assertEqual(t.load_credentials('https://models.example.test/custom/v1/','test-key','custom-model'),('test-key','https://models.example.test/custom/v1','custom-model'))
        self.assertEqual(t.load_credentials('http://localhost:1234/v1/chat/completions','test-key','local-model')[1],'http://localhost:1234/v1')
        with self.assertRaises(t.TranslationError):
            t.load_credentials('https://models.example.test','','custom-model')
        for base in ['file:///etc/passwd','https://user:password@example.test','https://example.test/v1?api_key=secret','https://example.test/#fragment','https://example.test:invalid']:
            with self.assertRaises(t.TranslationError):
                t.load_credentials(base,'test-key','custom-model')

    def test_custom_provider_payload_and_optional_parameter_fallback(self):
        requests=[]
        successful={'choices':[{'finish_reason':'stop','message':{'content':'{"translations":{"1":"你好"}}'}}],'usage':{'prompt_tokens':3,'completion_tokens':2}}
        class Opener:
            def open(self,request,timeout):
                body=json.loads(request.data)
                requests.append((request.full_url,body))
                if 'reasoning_effort' in body:
                    raise urllib.error.HTTPError(request.full_url,400,'bad',{},io.BytesIO(b'{"error":{"message":"unsupported parameter reasoning_effort"}}'))
                if 'response_format' in body:
                    raise urllib.error.HTTPError(request.full_url,400,'bad',{},io.BytesIO(b'{"error":{"message":"response_format not supported"}}'))
                if 'max_tokens' in body:
                    raise urllib.error.HTTPError(request.full_url,400,'bad',{},io.BytesIO(b'{"error":{"message":"max_tokens not supported, use max_completion_tokens"}}'))
                return io.BytesIO(json.dumps(successful).encode())
        api=t.API('test-key','https://models.example.test/custom/v1','custom-model',log=lambda s:None)
        api.opener=Opener()
        self.assertEqual(api.complete('Return JSON',{'items':{}})['translations']['1'],'你好')
        self.assertEqual(len(requests),4)
        self.assertEqual(requests[-1][0],'https://models.example.test/custom/v1/chat/completions')
        self.assertEqual(requests[-1][1]['model'],'custom-model')
        self.assertNotIn('thinking',requests[-1][1])
        api.complete('Return JSON',{'items':{}})
        self.assertEqual(len(requests),5)

    def test_api_redirect_does_not_forward_credentials(self):
        with self.assertRaises(t.TranslationError):
            t.NoRedirect().redirect_request(None,None,302,'redirect',{},'https://other.example.test')

    def test_compound_term_overrides_substring(self):
        self.assertEqual(t.matched_terms('ハイポーションを使う', {'ポーション':'药水','ハイポーション':'高级恢复剂'}), ['ハイポーション'])

    def test_job_lock_refuses_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'run.lock'
            with t.job_lock(p):
                with self.assertRaises(t.TranslationError):
                    with t.job_lock(p):
                        self.fail('concurrent writer admitted')

    def test_successful_item_survives_later_validation_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cache = t.Cache(Path(td)/'cache.db', 'scope')
            a = FakeAPI()
            def complete(system, payload, *args):
                return {'translations':{k:('休息' if v=='休む' else '@@P99@@') for k,v in payload['items'].items()}}
            a.complete = complete
            t.translate_group([(1,'休む'),(2,'村へ行く')],[],[],{},cache,a)
            self.assertEqual(cache.get('休む'), '休息')
            self.assertIsNone(cache.get('村へ行く'))
            cache.close()

    def test_real_main_compounds_are_not_forced_to_generic_translation(self):
        terms = t.effective_glossary({'メイン':'主要', 'ストーリー':'故事', 'メインストーリー1':'主线故事1', '移動（メイン街）':'移动（主城）'})
        self.assertEqual(terms['メインストーリー'], '主线故事')
        self.assertEqual(terms['メイン街'], '主城')
        t.validate_translation('メインストーリー2','主线故事2',terms)
        t.validate_translation('扉（メイン街）','门（主城）',terms)

    def test_no_partial_katakana_or_hiragana_constraints(self):
        self.assertEqual(t.matched_terms('マジックストーン',{'マジ':'魔法'}),[])
        self.assertEqual(t.matched_terms('はいった',{'はい':'是'}),[])
        self.assertEqual(t.matched_terms('ルナは休む',{'ルナ':'露娜'}),['ルナ'])

    def test_ambiguous_derived_compound_does_not_override_user(self):
        terms = {'移動（メイン街）':'移动（主城）','扉（メイン街）':'门（中心街）'}
        self.assertNotIn('メイン街',t.effective_glossary(terms))
        terms['メイン街']='主城区'
        self.assertEqual(t.effective_glossary(terms)['メイン街'],'主城区')

    def test_retry_contains_actual_reason_and_previous_translation(self):
        with tempfile.TemporaryDirectory() as td:
            cache=t.Cache(Path(td)/'cache.db','scope')
            a=FakeAPI()
            def complete(system,payload,*args):
                a.requests.append(payload)
                return {'translations':{'1':'露娜休息' if 'previous_validation_failures' in payload else '鲁娜休息'}}
            a.complete=complete
            result=t.translate_group([(1,'ルナは休む')],[],[],{'ルナ':'露娜'},cache,a)
            self.assertEqual(result,{'ルナは休む':'露娜休息'})
            repair=a.requests[1]['previous_validation_failures'][0]
            self.assertEqual(repair['questions'][0]['suggestion'],'露娜')
            self.assertEqual(repair['previous_translation'],'鲁娜休息')
            self.assertFalse(a.validation_issues)
            cache.close()

    def test_quarantined_item_does_not_stop_later_batches_and_can_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            t.write_json(root/'in.json',{'休む':'休む','村へ行く':'村へ行く'})
            (root/'out.work').mkdir()
            t.write_json(root/'out.work'/'glossary.json',{})
            a=FakeAPI()
            def complete(system,payload,*args):
                a.calls+=1
                return {'translations':{k:('@@P99@@' if v=='休む' else '前往村庄') for k,v in payload['items'].items()}}
            a.complete=complete
            report=t.run_translation(root/'in.json',root/'out.json',a,batch_size=1)
            self.assertFalse(report['complete'])
            self.assertEqual(report['validation_issue_count'],1)
            self.assertEqual(report['translated_unique'],1)
            self.assertFalse((root/'out.json').exists())
            partial=t.read_json(root/'out.partial.json')
            self.assertEqual(partial,{'休む':'休む','村へ行く':'前往村庄'})
            pending=t.read_json(root/'out.work'/'issues.json')['pending'][0]
            self.assertEqual(pending['source'],'休む')
            self.assertIn('占位符',pending['reason'])
            a2=FakeAPI()
            report=t.run_translation(root/'in.json',root/'out.json',a2,batch_size=1)
            self.assertTrue(report['complete'])
            self.assertEqual(a2.calls,1)
            self.assertEqual(t.read_json(root/'out.work'/'issues.json')['pending'],[])

    def test_invalid_legacy_cache_is_retranslated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            inp,out,_=self.setup_job(root)
            scope=t.digest([t.PROMPT_VERSION,FakeAPI.base,FakeAPI.model,{'ルナ':'露娜'},''])
            cache=t.Cache(root/'out.work'/'cache.sqlite3',scope)
            cache.put('村へ行く','坏@@P99@@')
            cache.close()
            a=FakeAPI()
            report=t.run_translation(inp,out,a)
            self.assertTrue(report['complete'])
            self.assertEqual(t.read_json(out)['村へ行く'],'前往村庄')

    def test_numbered_variables_may_reorder_but_format_controls_may_not(self):
        masked,parts=t.protect('%1は%2を奪った！')
        self.assertEqual(t.restore('从@@P1@@处夺取了@@P0@@！',parts),'从%2处夺取了%1！')
        t.validate_translation('%1は%2を奪った！','从%2处夺取了%1！',{})
        masked,parts=t.protect(r'\C[1]こんにちは\C[0]')
        with self.assertRaises(t.TranslationError):
            t.restore('@@P1@@你好@@P0@@',parts)
        with self.assertRaises(t.TranslationError):
            t.restore('@@P0@@你好',parts)

    def test_sense_overrides_do_not_silently_replace_cached_translations(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            terms={'バー':'条','ルナ':'露娜'}
            (root/'out.work').mkdir()
            t.write_json(root/'out.work'/'glossary.json',terms)
            t.write_json(root/'in.json',{'バー初回イベ':'バー初回イベ','HPバー非表示':'HPバー非表示','休む':'休む'})
            scope=t.digest([t.PROMPT_VERSION,FakeAPI.base,FakeAPI.model,terms,''])
            cache=t.Cache(root/'out.work'/'cache.sqlite3',scope)
            for source,target in [('バー初回イベ','条首次事件'),('HPバー非表示','隐藏HP条'),('休む','休息')]:
                cache.put(source,target)
            cache.close()
            t.write_json(root/'out.work'/'glossary-overrides.json',{'バー':'酒吧','HPバー':'HP条'})
            a=FakeAPI()
            def complete(system,payload,*args):
                a.calls+=1
                self.assertEqual(list(payload['items'].values()),['バー初回イベ'])
                return {'translations':{k:'酒吧首次事件' for k in payload['items']}}
            a.complete=complete
            report=t.run_translation(root/'in.json',root/'out.json',a)
            self.assertTrue(report['complete'])
            self.assertEqual(a.calls,0)
            self.assertEqual(t.read_json(root/'out.json')['HPバー非表示'],'隐藏HP条')

    def test_large_dictionary_key_integrity_offline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            game=root/'input.json'
            data={str(i):str(i) for i in range(1000)}
            t.write_json(game,data)
            original=game.read_bytes()
            work = root/'out.work'
            work.mkdir()
            t.write_json(work/'glossary.json', {})
            with patch.object(t, 'needs_translation', return_value=False):
                result = t.run_translation(game, root/'out.json', FakeAPI())
            output = t.read_json(root/'out.json')
            self.assertEqual(list(data), list(output))
            self.assertEqual(data, output)
            self.assertTrue(result['complete'])
            self.assertEqual(game.read_bytes(), original)

    def test_contextual_rebuttal_is_accepted_once_and_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'out.work').mkdir()
            t.write_json(root/'out.work'/'glossary.json',{'バー':'条'})
            t.write_json(root/'in.json',{'バー':'バー'})
            a=FakeAPI()
            def complete(system,payload,*args):
                a.calls+=1
                return {'translations':{'0':'酒吧'},'term_exceptions':{'0':[{'term':'バー','meaning':'提供饮酒的场所','rendering':'酒吧','reason':'当前条目是场所名称，不是界面计量条。'}]}}
            a.complete=complete
            report=t.run_translation(root/'in.json',root/'out.json',a)
            self.assertEqual(a.calls,1)
            self.assertEqual(t.read_json(root/'out.json')['バー'],'酒吧')
            self.assertEqual(report['semantic_exception_count'],1)
            self.assertEqual(report['semantic_warning_count'],0)
            a2=FakeAPI()
            report2=t.run_translation(root/'in.json',root/'out.json',a2)
            self.assertEqual(a2.calls,0)
            self.assertEqual(report2['semantic_exception_count'],1)

    def test_missing_rebuttal_is_queried_only_once_in_a_batch(self):
        with tempfile.TemporaryDirectory() as td:
            cache=t.Cache(Path(td)/'cache.db','scope')
            a=FakeAPI()
            def complete(system,payload,*args):
                a.calls+=1
                return {'translations':{ident:'酒馆' for ident in payload['items']}}
            a.complete=complete
            result=t.translate_group([(1,'バー初回'),(2,'バー終了')],[],[],{'バー':'条'},cache,a)
            self.assertEqual(a.calls,2)
            self.assertEqual(len(result),2)
            self.assertEqual(cache.decision('バー初回','酒馆')['status'],'needs_review')
            cache.close()

    def test_explanation_cannot_bypass_control_symbol_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cache=t.Cache(Path(td)/'cache.db','scope')
            a=FakeAPI()
            a.complete=lambda *args:{'translations':{'1':'酒吧'},'term_exceptions':{'1':[{'term':'バー','meaning':'场所','rendering':'酒吧','reason':'这里是酒吧'}]}}
            self.assertEqual(t.translate_group([(1,r'\C[1]バー')],[],[],{'バー':'条'},cache,a),{})
            self.assertIsNone(cache.get(r'\C[1]バー'))
            cache.close()

    def test_rebuttal_can_explain_compound_and_describe_rendering(self):
        decision=t.semantic_decision('メインストーリー','主线剧情',{'メイン':'主要'},[{'term':'メインストーリー','meaning':'主要剧情线','rendering':'按固定搭配译作主线剧情','reason':'这是复合词，不能将内部片段强制译为主要。'}])
        self.assertFalse(decision['unresolved'])
        self.assertEqual(decision['status'],'accepted_exception')

    def test_semantic_memory_is_relevant_and_bounded(self):
        a=FakeAPI()
        for i in range(20):
            t.remember_semantic_context(a,f'バー{i}',{'exceptions':[{'term':'バー','meaning':'酒吧','rendering':'酒吧','reason':'场所'}]})
        t.remember_semantic_context(a,'リング',{'exceptions':[{'term':'リング','meaning':'擂台','rendering':'擂台','reason':'竞技场所'}]})
        examples=t.relevant_semantic_context(a,[(1,'バー初回')])
        self.assertLessEqual(len(examples),2)
        self.assertTrue(all(entry['contextual_exceptions'][0]['term']=='バー' for entry in examples))


if __name__ == '__main__':
    unittest.main(verbosity=2)
