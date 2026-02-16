# 计划：Embedding 向量语义去重

## 背景

当前 synthesize Phase 2 使用字符级 3-gram Jaccard 相似度扫描重复对。
4,598 条知识单元在 50% 阈值下仅发现 1 对重复，因为 LLM 生成的文本措辞差异大，字面重叠极低。
需要替换为 Embedding 向量 + 余弦相似度，捕捉语义层面的重复。

## 当前架构（pipeline.py 关键位置）

- `SimilarityConfig` (line ~207): 相似度配置类，含 threshold / ngram_size
- `_compute_ngrams()` (line ~2507): 预计算 n-gram 集合
- `_jaccard_from_sets()` (line ~2516): Jaccard 相似度计算
- `scan_similarity()` (line ~2523): Phase 2 主函数，返回 intra_pairs + cross_pairs
- Phase 2 调用点 (line ~3014): `run_synthesize()` 中调用 `scan_similarity()`
- `KnowledgeUnit` 数据类: 含 fingerprint, title, body, topic 等字段
- 现有 API 调用基础设施: `_call_llm_api()`, `Endpoint`, `EndpointState`, failover 逻辑

## 实施步骤

### Step 1: 扩展配置

**文件**: `pipeline.py` (SimilarityConfig 类, line ~207)
**文件**: `config.yaml` (similarity 段)

在 `SimilarityConfig` 中新增字段:
```python
@dataclasses.dataclass
class SimilarityConfig:
    threshold: float = 0.70          # 保留，用于余弦相似度阈值（建议改默认 0.85）
    ngram_size: int = 3              # 保留，作为降级方案
    llm_batch_size: int = 8          # 保留
    max_file_size_kb: int = 100      # 保留
    method: str = "embedding"        # 新增: "embedding" | "ngram"
    embedding_model: str = "text-embedding-3-small"  # 新增
    embedding_batch_size: int = 100  # 新增: 每批 embedding 请求的单元数
```

config.yaml 对应新增:
```yaml
similarity:
  threshold: 0.85
  method: "embedding"
  embedding_model: "text-embedding-3-small"
  embedding_batch_size: 100
```

config 解析函数中（line ~336 附近）增加对新字段的读取。

### Step 2: 实现 Embedding 调用函数

**文件**: `pipeline.py`
**位置**: 在 `scan_similarity()` 函数之前新增

```python
def _get_embeddings(
    texts: List[str],
    config: PipelineConfig,
    batch_size: int = 100,
) -> List[List[float]]:
    """批量获取文本 embedding 向量。

    使用 synthesize 阶段的第一个端点（OpenAI 兼容格式）。
    端点 URL 需要支持 /embeddings 路径。
    """
```

实现要点:
- 复用现有端点配置（synthesize 端点），将 `/chat/completions` 替换为 `/embeddings`
- 批量请求（每批 100 条），避免单次请求过大
- 请求格式: `{"model": "text-embedding-3-small", "input": [texts...]}`
- 响应解析: `response["data"][i]["embedding"]`
- 错误处理: 复用现有 failover 逻辑或简单重试
- 返回与输入同序的向量列表

### Step 3: 实现 Embedding 缓存

**文件**: `pipeline.py`
**缓存路径**: `_intermediate/embedding_cache.json`

```python
def _load_embedding_cache(cache_path: Path) -> Dict[str, List[float]]:
    """加载 embedding 缓存。key = unit fingerprint, value = vector"""

def _save_embedding_cache(cache_path: Path, cache: Dict[str, List[float]]):
    """保存 embedding 缓存（原子写入）。"""
```

实现要点:
- key 使用 KnowledgeUnit.fingerprint（内容哈希）
- 增量友好：只对缓存中不存在的单元调用 API
- 使用现有的原子写入模式（写临时文件再 rename）
- 缓存文件预计大小: 4,598 × 1,536 维 × 8 字节 ≈ 54 MB JSON（考虑用 numpy .npy 或压缩）

**优化**: 如果 JSON 太大，改用二进制格式:
```python
import struct
# 存储: fingerprint_index.json (fp -> idx) + vectors.bin (连续 float32)
```

### Step 4: 实现余弦相似度扫描

**文件**: `pipeline.py`
**位置**: 新增函数，与 `scan_similarity()` 平级

```python
def scan_similarity_embedding(
    topic_units: Dict[str, List[KnowledgeUnit]],
    all_units: List[KnowledgeUnit],
    cached_fps: set,
    threshold: float,
    embeddings: Dict[str, List[float]],  # fingerprint -> vector
) -> Tuple[List[Tuple[KnowledgeUnit, KnowledgeUnit, float]],
           List[Tuple[KnowledgeUnit, KnowledgeUnit, float]]]:
    """Embedding 余弦相似度扫描。

    返回值与 scan_similarity() 相同:
    - intra_pairs: 同主题内超阈值的对 (送 LLM)
    - cross_pairs: 跨主题超阈值的对 (仅报告)
    """
```

实现要点:
- 用 numpy 矩阵运算批量计算余弦相似度（不要逐对 for 循环）
- 同主题内: 构建该主题所有单元的向量矩阵，计算相似度矩阵，提取上三角超阈值对
- 跨主题: 两个主题的向量矩阵做矩阵乘法
- 归一化向量后余弦相似度 = 点积，极快

```python
import numpy as np

def _cosine_matrix(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """计算两组向量的余弦相似度矩阵。"""
    # 归一化
    a_norm = vecs_a / np.linalg.norm(vecs_a, axis=1, keepdims=True)
    b_norm = vecs_b / np.linalg.norm(vecs_b, axis=1, keepdims=True)
    return a_norm @ b_norm.T
```

### Step 5: 修改 Phase 2 调用点

**文件**: `pipeline.py`
**位置**: `run_synthesize()` 函数中 Phase 2 段 (line ~3014)

```python
# Phase 2
if sim_cfg.method == "embedding":
    # 1. 加载缓存
    # 2. 找出未缓存的单元，批量调用 embedding API
    # 3. 保存缓存
    # 4. 调用 scan_similarity_embedding()
    intra_pairs, cross_pairs = scan_similarity_embedding(...)
else:
    # 降级: 原有 n-gram Jaccard 逻辑
    intra_pairs, cross_pairs = scan_similarity(...)
```

Phase 3/4/5/6 完全不改，intra_pairs / cross_pairs 格式与原来一致。

### Step 6: 添加 numpy 依赖

**执行**: `pip install numpy`（如果尚未安装）

numpy 是唯一新增依赖。embedding API 复用现有 HTTP 调用基础设施。

### Step 7: 调整 config.yaml 并测试

1. 先用 smoke 配置测试: 修改 `config.smoke.yaml` 添加 embedding 配置
2. 运行: `python pipeline.py --config config.smoke.yaml --only-stage synthesize --full-resync`
3. 验证日志输出 embedding 调用数、相似对数量
4. 确认后用生产配置运行

## 注意事项

1. **端点兼容性**: 确认 synthesize 端点支持 `/embeddings` API。如果不支持，需要单独配置 embedding 端点（如直接用 OpenAI 官方 API）
2. **阈值调优**: embedding 余弦相似度的 0.85 ≈ 语义高度相似，0.75 ≈ 主题相关。建议从 0.85 开始，根据 LLM 判断结果的 KEEP_BOTH 比例调整
3. **向量维度**: text-embedding-3-small 输出 1536 维。如果用其他模型注意维度变化
4. **内存**: 4,598 × 1536 float32 ≈ 27 MB，完全在内存范围内
5. **保留 n-gram 降级路径**: method="ngram" 时走原有逻辑，确保不破坏现有功能
6. **config.yaml 当前 threshold 是 0.50**: 实施完成后改回 0.85 用于 embedding 余弦相似度
