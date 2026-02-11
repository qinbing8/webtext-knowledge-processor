# 网文文档处理（`pipeline.py`）使用说明

本项目当前主流程是 `pipeline.py`：

- 阶段1：`extract`（逐文件去噪 + 知识提炼，输出每文件 Markdown）
- 阶段2：`synthesize`（按分类标签归组 → 逐分类 LLM 去重合并，输出分类级 Markdown）

---

## 1. 文件说明

- `pipeline.py`：双阶段管道主脚本（Prompt 不写死在代码中）
- `config.yaml`：正式配置——模型、密钥、系统提示词、输入输出路径、运行参数
- `config.smoke.yaml`：Smoke 测试配置（`_smoke_input` → `_smoke_output`）
- `系统提示词_模板.md`：系统提示词模板（供你复制到 `config.yaml`）
- `build_guide.py`：旧方案脚本（固定主题思路），仅作历史参考

---

## 2. 环境准备

### 2.1 Python 版本

建议 Python 3.10+。

```powershell
python -V
```

### 2.2 安装依赖

当前必需依赖：`PyYAML`

```powershell
python -m pip install pyyaml
```

---

## 3. 配置 `config.yaml`

编辑 `config.yaml`，重点确认这几项：

1. `api_keys`：填入对应厂商 API Key（不要保留占位文本）
2. `stages.extract`：阶段1的 `provider` + `model`
3. `stages.synthesize`：阶段2的 `provider` + `model`
4. `stages.extract.system_prompt`：阶段1系统提示词
5. `stages.synthesize.system_prompt`：阶段2系统提示词
6. `stages.extract.api_url`：阶段1 API 地址（可选，默认官方地址）
7. `stages.synthesize.api_url`：阶段2 API 地址（可选，默认官方地址）
8. `input_dir`：原始 `.txt` 目录
9. `output_dir`：输出目录

示例（按你需求可自由切换模型）：

```yaml
api_keys:
  gemini: "你的真实 key"
  claude: "你的真实 key"
  openai: "你的真实 key"

stages:
  extract:
    provider: "openai"
    model: "gpt-5.3-codex"
    api_url: "https://你的中转域名/v1/responses"
    system_prompt: |
      （在这里填写阶段1系统提示词，必须约束模型只输出 JSON）
  synthesize:
    provider: "claude"
    model: "claude-opus-4-6"
    api_url: "https://你的中转域名/v1/messages"
    system_prompt: |
      （在这里填写阶段2系统提示词）

input_dir: "D:/workspeace/网文文档处理"
output_dir: "D:/workspeace/网文文档处理/网文新人指南"
```

> 说明：`provider` 目前支持 `openai`、`claude`、`gemini`。
> `system_prompt` 建议使用 YAML 的 `|` 多行文本写法。
> `api_url` 可按阶段独立配置，用于第三方中转。

### 3.1 第三方中转地址说明（可选）

- OpenAI 阶段常见：`https://你的中转域名/v1/chat/completions`
- OpenAI 新接口也可用：`https://你的中转域名/v1/responses`
- Claude 阶段常见：`https://你的中转域名/v1/messages`
- Gemini 阶段可使用占位符：
  - `{model}`：会替换为 URL 编码后的模型名
  - `{api_key}`：会替换为 URL 编码后的密钥

Gemini 示例：

```yaml
api_url: "https://你的中转域名/v1beta/models/{model}:generateContent?key={api_key}"
```

若不填写 `api_url`，脚本自动使用官方默认地址。

补充：

- 当 `api_url` 是 `/v1/responses` 时，脚本会自动按 Responses API 的请求/响应格式处理。
- `api_keys` 支持逗号分隔多 Key，例如：`"key1,key2,key3"`，脚本会按顺序轮询尝试。

---

## 4. 运行方式

### 4.1 查看帮助

```powershell
python "pipeline.py" --help
```

### 4.2 执行全流程

```powershell
python "pipeline.py" --config "config.yaml" --only-stage all
```

### 4.3 只执行阶段1（extract）

```powershell
python "pipeline.py" --config "config.yaml" --only-stage extract
```

### 4.4 只执行阶段2（synthesize）

```powershell
python "pipeline.py" --config "config.yaml" --only-stage synthesize
```

> 注意：单独执行 `synthesize` 时，需要已存在 `output_dir/_intermediate/extract/*.md`。

### 4.5 Smoke 测试（小规模验证）

```powershell
python "pipeline.py" --config "config.smoke.yaml" --only-stage all
```

---

## 5. Synthesize 阶段工作原理

synthesize 阶段采用**先归组、后合并**的策略，将 extract 产出的细粒度知识单元归组为分类级文档：

### Phase 1：按分类标签归组（纯代码，零 LLM 调用）

- 读取 `_intermediate/extract/` 下所有 `.md` 文件
- 按 `## [分类标签][Px]` 格式的二级标题识别知识单元
- 相同分类标签的知识单元归入同一组
- 未匹配标签格式的单元归入"未分类"
- 输出：`{ 分类名: [知识单元文本, ...] }` 映射表

### Phase 2：逐分类 LLM 合并去重

- 遍历 Phase 1 的分类列表
- 每个分类只发送该分类下的所有知识单元
- 使用 `stages.synthesize.system_prompt` 作为系统提示词
- LLM 负责去重合并，保留所有独特内容
- 若分类下仅 1 条知识单元，直接写入无需调用 LLM
- 若分类内容超过批次大小限制，自动分批调用后拼接
- 进度显示：`[synthesize 2/6] 角色塑造 (15 条, 42.3KB)`

### Phase 3：写入与缓存

- 每个分类写入一个 `.md` 文件到 `topics/` 目录
- 文件名格式：`01_分类名.md`
- 写入缓存指纹和 manifest，重跑时如 extract 结果未变则自动跳过

---

## 6. 输出目录结构

默认在 `output_dir` 下生成：

```
output_dir/
├── _intermediate/
│   ├── extract/                          # 阶段1：每文件 extract Markdown
│   │   ├── 文件1.txt.md
│   │   └── 文件2.txt.md
│   ├── synthesize_fingerprint.txt        # synthesize 缓存指纹
│   ├── synthesize_manifest.json          # synthesize 缓存清单
│   └── run_summary.json                  # 运行摘要
├── topics/                               # 阶段2：分类级 Markdown 文档
│   ├── 01_角色塑造与代入感.md
│   ├── 02_剧情结构与节奏.md
│   └── ...
└── _logs/                                # 仅出错时生成
    └── pipeline_error_*.log
```

---

## 7. 断点续跑

- **extract 阶段**：已处理的文件会在 `_intermediate/extract/` 生成 Markdown，重跑时自动跳过已有结果。
- **synthesize 阶段**：基于 extract 指纹缓存，指纹不变时跳过。
- 如需完全重跑：删除 `output_dir` 整个目录再执行。

---

## 8. 常见问题

### 8.1 `python -m py_compile "pipeline.py"` 没有输出

这是正常现象。`py_compile` 在成功时通常不输出内容，报错时才显示异常。

### 8.2 报错"缺少依赖 PyYAML"

执行：

```powershell
python -m pip install pyyaml
```

### 8.3 报错"阶段配置错误：stages.*.system_prompt 不能为空"

说明你没有在 `config.yaml` 的对应阶段填写系统提示词。

### 8.4 401/403 鉴权失败

检查 `config.yaml` 中的 API Key 是否真实有效；不要保留"请填写..."占位文本。

### 8.5 报错"模型返回内容不是合法 JSON"

通常是 Phase 1 主题规划返回了非法 JSON。脚本会自动尝试 JSON 修复，如修复也失败则报此错。检查错误日志中的模型原始返回。

### 8.6 某个主题正文为空

Phase 2 为该主题生成正文时可能返回了空内容。重跑 synthesize 阶段即可：

```powershell
python "pipeline.py" --config "config.yaml" --only-stage synthesize
```

---

## 9. 旧脚本说明（`build_guide.py`）

`build_guide.py` 是旧版固定主题生成方案，不是当前主流程。

- 你现在的主流程：`pipeline.py`
- `build_guide.py` 建议仅用于历史对照或回溯

---
