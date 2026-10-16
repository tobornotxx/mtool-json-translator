# MTool 文本翻译器 v1.0

一个纯 **Vibe Coding** 的个人小项目：读取 MTool / MTools 导出的文本 JSON，保持原文键名不变，把对应的值翻译成简体中文，生成可重新导入翻译替换工具的 JSON 文件。

这个仓库主要用于备份、存档自己写过的代码和软件。Windows EXE 放在 [Releases](https://github.com/tobornotxx/mtool-json-translator/releases) 中下载。欢迎有类似需求的朋友试用，并通过 Issues 提交使用体验、问题或建议。

## 它做什么

输入是一个“原文 → 文本值”的 JSON 对象。例如：

```json
{
  "こんにちは。": "こんにちは。",
  "アイテム": "アイテム"
}
```

输出仍保留相同的键，只把值替换成中文译文：

```json
{
  "こんにちは。": "你好。",
  "アイテム": "物品"
}
```

不会覆盖原始输入文件。游戏资源、游戏文本和个人翻译缓存不包含在仓库或安装包内。

## 下载与使用

1. 在 [Releases](https://github.com/tobornotxx/mtool-json-translator/releases) 下载 Windows x64 EXE，双击运行；无需安装 Python。
2. 选择 MTool 导出的 JSON，确认新的输出文件路径。
3. 输入自己的 **API Base、API Key、Model Name**。
4. 可以先生成并检查术语建议，再开始翻译。
5. 完成后，把输出 JSON 加载到自己的翻译替换工具中，并在游戏中确认显示效果。

输入和输出必须是扁平的 JSON 字符串映射，不支持把任意嵌套 JSON 或其他字幕格式直接当作词典使用。

## 模型配置

支持提供 **OpenAI 兼容 Chat Completions 接口**的服务或本地模型网关。API Base 可以包含 `/v1` 或服务商要求的其他路径；程序会调用其下的 `/chat/completions`。

- **API Base**：不填写时，灰色提示为 `https://api.deepseek.com`，并使用这个默认值。
- **Model Name**：不填写时，灰色提示为 `deepseek-flash`，并使用这个默认值。
- **API Key**：必须由用户填写，仅在当前会话的内存中使用，不写入配置、日志或翻译缓存。

这些只是留空时的默认预设，不限制只能使用某一家服务。换接口时，同时填写对应的 Base、Key 和模型名即可。程序仅使用用户提供的配置，不内置任何个人密钥或游戏目录。

部分兼容接口不支持 JSON 模式、温度或推理设置，程序会在接口明确拒绝对应参数时移除该可选参数，再尝试一次；必要时适配 `max_completion_tokens`。各家原生私有协议、图片接口、Responses API 等不在本工具的兼容范围内。

## 翻译与术语

- 相邻文本分批翻译，附带少量前后文，不把整个 JSON 塞进一个请求。
- 先抽取候选术语和分散样本，形成术语建议及背景说明。
- 同一实体和同一词义尽量统一；**术语表是建议，不是强制字符串替换规则**。
- AI 可以返回当前词义、实际译法及例外理由。理由与译文一起缓存，续译不会重复拦截已接受的例外。
- 如果检测出差异但没有解释，只合并询问一次；允许 AI 说明原译正确，不为了匹配建议词而强改。
- 控制符、变量数量、条目对应、空值等格式问题仍严格检查。编号变量允许根据中文语序移动，颜色等格式控制符保留顺序。
- 本地缓存和断点续译减少重复请求；服务商是否支持前缀缓存及如何计费，由对应服务决定。

语义判断仍依赖模型；上下文可能不完整，术语和例外理由也可能有误。请以游戏内语境为准。这是个人使用和备份项目，并未覆盖所有游戏与接口。

## 查看例外理由

接受术语例外时，窗口日志直接显示 **原文、AI 译文、当前词义、本条译法和 AI 理由**。

翻译完成后，点击“查看已保存的术语例外”，可以读取已有记录。这个操作只读本地文件，不重新翻译、不调用 API。

## 输出与缓存

默认输出为输入文件旁的 `<原文件名>.zh-CN.json`。未完成时保存为 `<原文件名>.zh-CN.partial.json`。

同名 `.work` 目录保存：

- `glossary.json`：术语建议。
- `glossary-overrides.json`：可选的补充建议。
- `context.json`：背景和语气说明。
- `cache.sqlite3`：译文及例外理由的缓存。
- `semantic-decisions.json`：例外和语义提示。
- `issues.json`：格式校验失败的条目。
- `report.json`：进度与接口返回的用量统计。

停止时点击“停止并保存”，等待当前请求结束后再关闭。再次选择同一输入、输出和模型配置即可续译。原始 JSON 的结构决定，同一个键只能对应一个译文，无法同时表达不同场景中同一句原文的多种译法。

缓存和日志可能包含你自己的游戏文本，请保留在本地，不要将它们上传到公开仓库。API Key 只会发送给你填写的 API Base，请确认地址正确；程序不会向重定向地址转发密钥。

## 从源码运行

运行环境：Python 3.10+，含 Tkinter。翻译器运行时只使用标准库。

```console
python translator.py
python -m unittest -v tests
```

命令行模式从环境变量读取密钥，避免将密钥放入命令行参数：

```powershell
$env:MTOOL_TRANSLATOR_API_KEY = Read-Host 'API Key'
python translator.py --input "ManualTransFile.json" --output "translated.json" --api-base "https://your-provider.example/v1" --model "your-model" --log "translation.log"
Remove-Item Env:MTOOL_TRANSLATOR_API_KEY
```

`--glossary-only` 只生成术语建议；`--limit 20` 仅处理 20 条未缓存文本。省略 Base 和模型时使用上述默认预设。

## 打包 EXE

在 Windows 上运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-build.txt
.\.venv\Scripts\python -m PyInstaller --noconfirm --onefile --windowed --name MToolTranslator --version-file version_info.txt --distpath dist --workpath build --specpath build translator.py
```

生成文件位于 `dist/MToolTranslator.exe`。打包产物通过 GitHub Releases 分发，不提交到 Git 源码历史。

## 反馈

欢迎提 Issue，说明输入文件的结构、所用兼容接口和复现步骤。请不要附上 API Key、个人配置或无权公开的完整游戏文本。
