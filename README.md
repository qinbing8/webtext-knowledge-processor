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

synthesize 阶段采用**六阶段流水线**，将 extract 产出的细粒度知识单元归组、去重、合并为分类级文档：

### Phase 1：结构化解析 + 归组（纯代码，零 LLM 调用）

- 读取 `_intermediate/extract/` 下所有 `.md` 文件
- 按 `## [分类标签][Px]` 格式的二级标题识别知识单元
- 通过 `config.yaml` 中的 `mapping` 规则将标签映射到主题
- 构建 unit 指纹缓存，识别新增/已缓存单元
- 输出：`{ 主题名: [KnowledgeUnit, ...] }` 映射表

### Phase 2：增量相似度扫描

- 支持两种模式：**embedding 余弦相似度**（默认）或 **n-gram Jaccard** 降级
- 仅对新增单元与全量单元进行比较，已缓存的对跳过
- 产出同主题内相似对（`intra_pairs`）和跨主题相似对（`cross_pairs`）

### Phase 3：LLM 判断 + 合并（两阶段）

- **Phase 3a（verdict）**：将相似度 ≥ 阈值的同主题对批量发送 LLM，判断 DUPLICATE / MERGE / KEEP_BOTH
- **Phase 3b（merge）**：对判定为 MERGE 的对逐对调用 LLM 生成合并文本
- 判断结果持久化缓存，重跑时自动复用

### Phase 4：内存中去重/合并

- 根据 Phase 3 的判断结果，在内存中执行去重和合并操作
- 输出去重后的 `deduped_units`

### Phase 5：拼接写入文件（含分卷）

- 按主题写入 `topics/` 目录下的 `.md` 文件
- 单文件超过分卷阈值（默认 100KB）时自动按知识单元边界切分为多卷
- 内容未变的主题自动跳过写入（指纹缓存命中）

### Phase 6：生成目录与报告

- 生成 `00-目录与导读.md`（主题总览表）
- 生成 `cross_topic_similarity.txt`（跨主题相似度报告）
- 生成 `changelog_{timestamp}.md`（增量变更报告，首次运行不生成）

### 增量变更报告

每次增量运行后，在 `_intermediate/` 目录自动生成一份变更报告，列出本次新增的知识单元及其所在文件，按主题分组排序。首次运行（无旧缓存）不生成报告。

报告示例：

```markdown
# Synthesize 变更报告

> 生成时间: 2026-02-19 14:30:00
> 新增知识单元: 47 条

## 08-创作心态篇 (+12)

- [08-创作心态篇.md](../topics/08-创作心态篇.md) — [写作心态][P2] 自我审视能力的前提是审美
- ...
```

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
│   ├── unit_fingerprints.json            # 知识单元指纹缓存
│   ├── topic_fingerprints.json           # 主题内容指纹缓存
│   ├── embedding_cache.json              # embedding 向量缓存
│   ├── cross_topic_similarity.txt        # 跨主题相似度报告
│   ├── changelog_20260219_143000.md      # 增量变更报告（每次运行生成）
│   └── run_summary.json                  # 运行摘要
├── topics/                               # 阶段2：主题级 Markdown 文档
│   ├── 00-目录与导读.md
│   ├── 01-认知与定位篇.md
│   ├── 02-选题与开篇篇.md
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
