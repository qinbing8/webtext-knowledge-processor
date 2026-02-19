from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import ssl
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from urllib import error, parse, request

import numpy as np


IGNORED_DIR_NAMES = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
}


class PipelineError(RuntimeError):
    """管道执行异常。"""


class EndpointsExhaustedError(RuntimeError):
    """所有端点均已熔断，需要用户检查中转站配置。"""


@dataclass(frozen=True)
class Endpoint:
    """单个 API 端点（中转站）配置。"""
    name: str
    api_url: str
    api_key: str
    api_type: str = ""  # 可选覆盖，为空则 fallback 到 StageConfig.api_type
    model: str = ""     # 可选覆盖，为空则 fallback 到 StageConfig.model


@dataclass(frozen=True)
class StageConfig:
    provider: str
    model: str
    system_prompt: str
    api_type: str
    api_url: str | None = None
    api_key: str | None = None
    endpoints: tuple = ()  # Tuple[Endpoint, ...] 多端点故障转移列表

    def __repr__(self) -> str:
        if self.endpoints:
            ep_info = f"endpoints={len(self.endpoints)}"
        else:
            masked_key = "***" if self.api_key else None
            ep_info = f"api_url={self.api_url!r}, api_key={masked_key!r}"
        return (
            f"StageConfig(provider={self.provider!r}, model={self.model!r}, "
            f"system_prompt=<{len(self.system_prompt)} chars>, "
            f"api_type={self.api_type!r}, {ep_info})"
        )


# ── 熔断参数 ──
CIRCUIT_FAILURE_THRESHOLD = 3    # 连续失败 N 次触发熔断
CIRCUIT_COOLDOWN_SECONDS = 18000  # 熔断冷却期（5 小时）


class EndpointState:
    """端点运行时状态（熔断管理）。

    在一次 pipeline 运行中，每个端点有一个 EndpointState 实例，
    跟踪连续失败次数并在达到阈值时触发熔断。
    """

    def __init__(self, endpoint: Endpoint, log_dir: Path | None = None) -> None:
        self.endpoint = endpoint
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0
        self.log_dir = log_dir
        self._last_errors: List[str] = []
        self._lock = threading.Lock()  # 线程安全：保护熔断状态读写

    @property
    def name(self) -> str:
        return self.endpoint.name

    def is_available(self) -> bool:
        with self._lock:
            if self.consecutive_failures < CIRCUIT_FAILURE_THRESHOLD:
                return True
            return time.time() >= self.circuit_open_until

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.circuit_open_until = 0.0

    def record_failure(self, error_msg: str = "") -> None:
        with self._lock:
            self.consecutive_failures += 1
            if error_msg:
                self._last_errors.append(error_msg)
            if self.consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
                self.circuit_open_until = time.time() + CIRCUIT_COOLDOWN_SECONDS
                cooldown_hours = CIRCUIT_COOLDOWN_SECONDS / 3600
                print(
                    f"\n{'='*60}\n"
                    f"[failover] 端点 '{self.name}' 连续失败 "
                    f"{self.consecutive_failures} 次，熔断 {cooldown_hours:.1f} 小时\n"
                    f"[failover] 请检查上游端点配置（模型名称、API Key、服务状态）\n"
                    f"{'='*60}"
                )
                self._write_circuit_log()

    def _write_circuit_log(self) -> None:
        """熔断时写入详细日志文件。"""
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"circuit_break_{self.name}_{timestamp}.log"
        cooldown_hours = CIRCUIT_COOLDOWN_SECONDS / 3600
        lines = [
            f"时间: {dt.datetime.now().isoformat(timespec='seconds')}",
            f"端点: {self.name}",
            f"API URL: {sanitize_url_for_log(self.endpoint.api_url)}",
            f"模型: {self.endpoint.model or '(继承 stage 配置)'}",
            f"API 类型: {self.endpoint.api_type or '(继承 stage 配置)'}",
            f"连续失败次数: {self.consecutive_failures}",
            f"熔断冷却: {cooldown_hours:.1f} 小时",
            "",
            "最近错误:",
        ]
        for i, err in enumerate(self._last_errors[-5:], 1):
            lines.append(f"  {i}. {err[:500]}")
        lines.extend([
            "",
            "建议排查:",
            "  1. 检查端点是否支持配置的模型名称",
            "  2. 检查 API Key 是否有效",
            "  3. 检查上游服务是否正常运行",
            "  4. 确认模型名称拼写正确（如 claude-opus-4-6 vs claude-opus-4-6-thinking）",
        ])
        log_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[failover] 熔断日志已写入：{log_path}")


@dataclass(frozen=True)
class RuntimeConfig:
    request_timeout_seconds: int = 120
    max_retries: int = 2
    retry_backoff_seconds: int = 3


@dataclass(frozen=True)
class ProviderLimits:
    gemini: int = 100    # KB
    claude: int = 353    # KB
    openai: int = 150    # KB


@dataclass(frozen=True)
class ExtractBatchingConfig:
    enabled: bool = True
    max_files_per_batch: int = 50
    max_batch_size_kb: int = 353


@dataclass(frozen=True)
class SynthesizeBatchingConfig:
    max_batch_size_kb: int = 353


@dataclass(frozen=True)
class ConcurrencyConfig:
    """并发配置：控制 ThreadPoolExecutor 的工作线程数。"""
    max_workers: int = 3  # 并发 LLM 调用数


@dataclass(frozen=True)
class BatchingConfig:
    provider_limits: ProviderLimits = field(default_factory=ProviderLimits)
    extract: ExtractBatchingConfig = field(default_factory=ExtractBatchingConfig)
    synthesize: SynthesizeBatchingConfig = field(default_factory=SynthesizeBatchingConfig)


@dataclass
class KnowledgeUnit:
    """单条知识单元的结构化表示。"""
    tag: str           # 分类标签（如"金三章"）
    priority: int      # 0=P0, 1=P1, 2=P2, 3=无标注
    title: str         # 知识点标题
    body: str          # 完整文本（含 ## 标题行）
    fingerprint: str   # 内容指纹（sha256[:16]）


@dataclass(frozen=True)
class SimilarityConfig:
    """相似度扫描配置。"""
    threshold: float = 0.70
    ngram_size: int = 3
    llm_batch_size: int = 8
    max_file_size_kb: int = 100
    method: str = "embedding"              # "embedding" | "ngram"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    embedding_api_url: str = ""            # 可选：独立 embedding 端点 URL
    embedding_api_key: str = ""            # 可选：独立 embedding 端点 Key


@dataclass(frozen=True)
class SynthesizeMergeConfig:
    mapping: Dict[str, List[str]] = field(default_factory=dict)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)


@dataclass(frozen=True)
class PipelineConfig:
    api_keys: Dict[str, str]
    extract_stage: StageConfig
    synthesize_stage: StageConfig
    input_dir: Path
    output_dir: Path
    runtime: RuntimeConfig
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    merge: SynthesizeMergeConfig = field(default_factory=SynthesizeMergeConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="纯 LLM 双阶段文档处理管道")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径（默认：config.yaml）",
    )
    parser.add_argument(
        "--only-stage",
        choices=["all", "extract", "synthesize"],
        default="all",
        help="只执行某个阶段（默认：all）",
    )
    parser.add_argument(
        "--full-resync",
        action="store_true",
        default=False,
        help="强制全量重新合成（忽略增量缓存）",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> PipelineConfig:
    try:
        import yaml  # 延迟导入，确保 --help 不依赖该包
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("缺少依赖 PyYAML，请先执行：python -m pip install pyyaml") from exc

    if not config_path.is_file():
        raise PipelineError(f"配置文件不存在：{config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PipelineError("配置文件格式错误：顶层必须为映射")

    api_keys = raw.get("api_keys")
    stages = raw.get("stages")
    input_dir = raw.get("input_dir")
    output_dir = raw.get("output_dir")
    runtime_raw = raw.get("runtime", {})

    if not isinstance(api_keys, dict):
        raise PipelineError("配置项 api_keys 缺失或格式错误")
    if not isinstance(stages, dict):
        raise PipelineError("配置项 stages 缺失或格式错误")
    if not isinstance(input_dir, str) or not input_dir.strip():
        raise PipelineError("配置项 input_dir 缺失或为空")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise PipelineError("配置项 output_dir 缺失或为空")

    extract_stage = build_stage_config("extract", stages.get("extract"))
    synthesize_stage = build_stage_config("synthesize", stages.get("synthesize"))

    runtime = RuntimeConfig(
        request_timeout_seconds=int(runtime_raw.get("request_timeout_seconds", 120)),
        max_retries=int(runtime_raw.get("max_retries", 2)),
        retry_backoff_seconds=int(runtime_raw.get("retry_backoff_seconds", 3)),
    )

    # ── 解析 batching 配置 ──
    batching_raw = raw.get("batching", {}) or {}
    provider_limits_raw = batching_raw.get("provider_limits", {}) or {}
    extract_batch_raw = batching_raw.get("extract", {}) or {}
    synthesize_batch_raw = batching_raw.get("synthesize", {}) or {}

    provider_limits = ProviderLimits(
        gemini=int(provider_limits_raw.get("gemini", 100)),
        claude=int(provider_limits_raw.get("claude", 200)),
        openai=int(provider_limits_raw.get("openai", 150)),
    )
    for _name in ("gemini", "claude", "openai"):
        if getattr(provider_limits, _name) <= 0:
            raise PipelineError(f"配置项 batching.provider_limits.{_name} 必须为正数")

    extract_max_batch_kb = int(extract_batch_raw.get("max_batch_size_kb", 200))
    if extract_max_batch_kb <= 0:
        raise PipelineError("配置项 batching.extract.max_batch_size_kb 必须为正数")

    synthesize_max_batch_kb = int(synthesize_batch_raw.get("max_batch_size_kb", 200))
    if synthesize_max_batch_kb <= 0:
        raise PipelineError("配置项 batching.synthesize.max_batch_size_kb 必须为正数")

    batching = BatchingConfig(
        provider_limits=provider_limits,
        extract=ExtractBatchingConfig(
            enabled=bool(extract_batch_raw.get("enabled", True)),
            max_files_per_batch=int(extract_batch_raw.get("max_files_per_batch", 50)),
            max_batch_size_kb=extract_max_batch_kb,
        ),
        synthesize=SynthesizeBatchingConfig(
            max_batch_size_kb=synthesize_max_batch_kb,
        ),
    )

    # ── 解析 synthesize.grouping 配置 ──
    synthesize_raw = stages.get("synthesize", {}) or {}
    grouping_raw = synthesize_raw.get("grouping", {}) or {}
    grouping_mapping_raw = grouping_raw.get("mapping", {}) or {}
    grouping_mapping: Dict[str, List[str]] = {}
    for cat_name, topics in grouping_mapping_raw.items():
        if isinstance(topics, list):
            grouping_mapping[str(cat_name)] = [str(t) for t in topics]

    # ── 解析 synthesize.similarity 配置 ──
    similarity_raw = synthesize_raw.get("similarity", {}) or {}
    sim_threshold = float(similarity_raw.get("threshold", 0.70))
    sim_ngram = int(similarity_raw.get("ngram_size", 3))
    sim_batch = int(similarity_raw.get("llm_batch_size", 8))
    sim_max_kb = int(similarity_raw.get("max_file_size_kb", 100))
    sim_method = str(similarity_raw.get("method", "embedding")).strip().lower()
    sim_embedding_model = str(similarity_raw.get("embedding_model", "text-embedding-3-small")).strip()
    sim_embedding_batch = int(similarity_raw.get("embedding_batch_size", 100))
    # 可选独立 embedding 端点
    emb_endpoint_raw = similarity_raw.get("embedding_endpoint", {}) or {}
    sim_emb_api_url = str(emb_endpoint_raw.get("api_url", "")).strip()
    sim_emb_api_key = str(emb_endpoint_raw.get("api_key", "")).strip()
    if not (0 < sim_threshold <= 1):
        raise PipelineError(f"similarity.threshold 必须在 (0, 1] 范围内，当前值: {sim_threshold}")
    if sim_ngram < 2:
        raise PipelineError(f"similarity.ngram_size 必须 >= 2，当前值: {sim_ngram}")
    if sim_batch < 1:
        raise PipelineError(f"similarity.llm_batch_size 必须 >= 1，当前值: {sim_batch}")
    if sim_max_kb < 1:
        raise PipelineError(f"similarity.max_file_size_kb 必须 >= 1，当前值: {sim_max_kb}")
    if sim_method not in ("embedding", "ngram"):
        raise PipelineError(f"similarity.method 必须为 'embedding' 或 'ngram'，当前值: {sim_method!r}")
    if sim_embedding_batch < 1:
        raise PipelineError(f"similarity.embedding_batch_size 必须 >= 1，当前值: {sim_embedding_batch}")
    similarity_config = SimilarityConfig(
        threshold=sim_threshold,
        ngram_size=sim_ngram,
        llm_batch_size=sim_batch,
        max_file_size_kb=sim_max_kb,
        method=sim_method,
        embedding_model=sim_embedding_model,
        embedding_batch_size=sim_embedding_batch,
        embedding_api_url=sim_emb_api_url,
        embedding_api_key=sim_emb_api_key,
    )

    merge_config = SynthesizeMergeConfig(
        mapping=grouping_mapping,
        similarity=similarity_config,
    )

    # ── 解析 concurrency 配置 ──
    concurrency_raw = raw.get("concurrency", {}) or {}
    concurrency = ConcurrencyConfig(
        max_workers=max(1, int(concurrency_raw.get("max_workers", 3))),
    )

    return PipelineConfig(
        api_keys={str(k).lower(): str(v) for k, v in api_keys.items()},
        extract_stage=extract_stage,
        synthesize_stage=synthesize_stage,
        input_dir=Path(input_dir).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        runtime=runtime,
        batching=batching,
        merge=merge_config,
        concurrency=concurrency,
    )


def build_stage_config(stage_name: str, raw_stage: Any) -> StageConfig:
    if not isinstance(raw_stage, dict):
        raise PipelineError(f"阶段配置缺失：stages.{stage_name}")

    provider = raw_stage.get("provider")
    model = raw_stage.get("model")
    system_prompt = raw_stage.get("system_prompt")
    api_type = raw_stage.get("api_type")
    api_url = raw_stage.get("api_url", "")
    api_key_raw = raw_stage.get("api_key") or raw_stage.get("api_keys")
    if not isinstance(provider, str) or not provider.strip():
        raise PipelineError(f"阶段配置错误：stages.{stage_name}.provider 不能为空")
    if not isinstance(model, str) or not model.strip():
        raise PipelineError(f"阶段配置错误：stages.{stage_name}.model 不能为空")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise PipelineError(f"阶段配置错误：stages.{stage_name}.system_prompt 不能为空")
    if api_type is not None and not isinstance(api_type, str):
        raise PipelineError(f"阶段配置错误：stages.{stage_name}.api_type 必须是字符串")
    if not isinstance(api_url, str):
        raise PipelineError(f"阶段配置错误：stages.{stage_name}.api_url 必须是字符串")

    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"openai", "claude", "gemini"}:
        raise PipelineError(
            f"不支持的 provider：{provider}（仅支持 openai / claude / gemini）"
        )

    default_api_type_map = {
        "openai": "openai",
        "claude": "anthropic",
        "gemini": "gemini",
    }
    normalized_api_type = (api_type.strip().lower() if isinstance(api_type, str) else "") or default_api_type_map[normalized_provider]
    if normalized_api_type not in {"openai", "anthropic", "gemini"}:
        raise PipelineError(
            f"阶段配置错误：stages.{stage_name}.api_type 仅支持 openai / anthropic / gemini"
        )

    normalized_api_url = api_url.strip() or None
    if normalized_api_url and not re.match(r"^https?://", normalized_api_url):
        raise PipelineError(
            f"阶段配置错误：stages.{stage_name}.api_url 必须是 http/https 地址"
        )

    normalized_api_key = str(api_key_raw).strip() if api_key_raw is not None else None
    if normalized_api_key is not None and not normalized_api_key:
        normalized_api_key = None

    # ── 解析 endpoints 列表（新格式） ──
    endpoints_raw = raw_stage.get("endpoints")
    endpoints: List[Endpoint] = []
    if isinstance(endpoints_raw, list) and endpoints_raw:
        for i, ep_raw in enumerate(endpoints_raw):
            if not isinstance(ep_raw, dict):
                raise PipelineError(
                    f"阶段配置错误：stages.{stage_name}.endpoints[{i}] 必须是映射"
                )
            ep_name = str(ep_raw.get("name", f"endpoint_{i + 1}"))
            ep_url = str(ep_raw.get("api_url", "")).strip()
            ep_key = str(ep_raw.get("api_key", "")).strip()
            if not ep_url:
                raise PipelineError(
                    f"阶段配置错误：stages.{stage_name}.endpoints[{i}].api_url 不能为空"
                )
            if not ep_key:
                raise PipelineError(
                    f"阶段配置错误：stages.{stage_name}.endpoints[{i}].api_key 不能为空"
                )
            if not re.match(r"^https?://", ep_url):
                raise PipelineError(
                    f"阶段配置错误：stages.{stage_name}.endpoints[{i}].api_url 必须是 http/https 地址"
                )
            ep_api_type = str(ep_raw.get("api_type", "")).strip().lower()
            if ep_api_type and ep_api_type not in {"openai", "anthropic", "gemini"}:
                raise PipelineError(
                    f"阶段配置错误：stages.{stage_name}.endpoints[{i}].api_type "
                    f"仅支持 openai / anthropic / gemini"
                )
            ep_model = str(ep_raw.get("model", "")).strip()
            endpoints.append(Endpoint(name=ep_name, api_url=ep_url, api_key=ep_key, api_type=ep_api_type, model=ep_model))

    return StageConfig(
        provider=normalized_provider,
        model=model.strip(),
        system_prompt=system_prompt.strip(),
        api_type=normalized_api_type,
        api_url=normalized_api_url,
        api_key=normalized_api_key,
        endpoints=tuple(endpoints),
    )


# ── 文件工具 ──


def list_source_files(input_dir: Path, output_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise PipelineError(f"输入目录不存在：{input_dir}")

    files: List[Path] = []
    for path in input_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        parts_lower = {part.lower() for part in path.parts}
        if any(name.lower() in parts_lower for name in IGNORED_DIR_NAMES):
            continue
        if output_dir == path or output_dir in path.parents:
            continue
        files.append(path)

    files.sort(key=lambda p: p.relative_to(input_dir).as_posix().lower())
    return files


def read_text_auto(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    if not data:
        return "", "utf-8"

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16"), "utf-16-le(bom)"
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16-be(bom)"

    candidates = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gb18030",
        "gbk",
        "big5",
    ]
    for encoding in candidates:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return data.decode("gb18030", errors="replace"), "gb18030(replace)"


def safe_filename(name: str) -> str:
    value = name.strip().replace("\\", "/")
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace(" ", "_")
    value = value.strip("._")
    return value or "untitled"


def _batch_output_name(names: List[str]) -> str:
    """基于批次内文件名列表生成确定性文件名。

    使用文件名排序后的 SHA-256 前 8 位，确保同一组文件
    无论运行顺序如何，始终映射到相同的输出文件名。
    避免断点续传时因批次重编号而覆盖已完成批次的结果。
    """
    key = "|".join(sorted(names))
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"batch_{h}.md"


# ── API Key 工具 ──


def parse_api_keys(raw: str) -> List[str]:
    parts = [item.strip() for item in re.split(r"[,，]", raw) if item.strip()]
    return parts


def get_api_keys(config: PipelineConfig, provider: str, stage_api_key: str | None = None) -> List[str]:
    # 优先使用阶段级 api_key
    if stage_api_key:
        keys = parse_api_keys(stage_api_key)
        if keys:
            return keys

    from_config = config.api_keys.get(provider, "").strip()
    if from_config:
        keys = parse_api_keys(from_config)
        if keys:
            return keys

    env_map = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_name = env_map[provider]
    from_env = os.getenv(env_name, "").strip()
    if from_env:
        keys = parse_api_keys(from_env)
        if keys:
            return keys
    raise PipelineError(f"未找到 {provider} 的 API Key，请在 config.yaml 或环境变量中配置")


# ── 批处理工具函数 ──


def measure_payload_size_kb(obj: Any) -> float:
    """JSON 序列化后 UTF-8 字节数 / 1024。"""
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) / 1024.0


def get_effective_size_limit(
    soft_limit_kb: float,
    provider: str,
    provider_limits: ProviderLimits,
    safety_margin: float = 0.85,
) -> float:
    """返回 min(软限制, provider 硬限制) * safety_margin。

    safety_margin 用于预留 prompt 包装、JSON 结构等请求开销，
    避免实际 payload 接近 provider 硬上限时触发 413/429。
    """
    hard_limits = {
        "gemini": provider_limits.gemini,
        "claude": provider_limits.claude,
        "openai": provider_limits.openai,
    }
    hard_limit = hard_limits.get(provider.lower(), 200)
    return min(soft_limit_kb, hard_limit) * safety_margin


def compute_batches(
    items: List[Any],
    max_items: int,
    max_size_kb: float,
) -> List[List[Any]]:
    """顺序贪心装箱，双重约束（文件数 + 总大小）。

    调用方应预先对 items 排序（如按大小升序）。
    单个 item 超限时作为独立批次并打印警告。
    """
    if not items:
        return []

    batches: List[List[Any]] = []
    current_batch: List[Any] = []
    current_size_kb = 0.0

    for item in items:
        item_size = measure_payload_size_kb(item)

        # 单个 item 就超限 → 作为独立批次
        if item_size > max_size_kb:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size_kb = 0.0
            print(f"[batching] 警告：单个条目 {item_size:.1f}KB 超过批次上限 {max_size_kb}KB，独立成批")
            batches.append([item])
            continue

        # 加入当前批次会超限 → 封装当前批次、开新批次
        if current_batch and (
            len(current_batch) >= max_items or current_size_kb + item_size > max_size_kb
        ):
            batches.append(current_batch)
            current_batch = []
            current_size_kb = 0.0

        current_batch.append(item)
        current_size_kb += item_size

    if current_batch:
        batches.append(current_batch)

    return batches


# ── HTTP / LLM 调用 ──


def normalize_openai_url(api_url: str | None) -> str:
    if not api_url:
        return "https://api.openai.com/v1/chat/completions"

    split = parse.urlsplit(api_url)
    path = split.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return api_url
    if path.endswith("/responses"):
        return api_url

    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif "/v1/" in path:
        new_path = f"{path}/chat/completions"
    elif not path:
        new_path = "/v1/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions"
    return parse.urlunsplit((split.scheme, split.netloc, new_path, split.query, split.fragment))


def normalize_anthropic_url(api_url: str | None) -> str:
    if not api_url:
        return "https://api.anthropic.com/v1/messages"

    split = parse.urlsplit(api_url)
    path = split.path.rstrip("/")
    if path.endswith("/messages"):
        return api_url

    if path.endswith("/v1"):
        new_path = f"{path}/messages"
    elif not path:
        new_path = "/v1/messages"
    else:
        new_path = f"{path}/messages"
    return parse.urlunsplit((split.scheme, split.netloc, new_path, split.query, split.fragment))


def normalize_gemini_url(api_url: str | None, model: str, api_key: str) -> str:
    encoded_model = parse.quote(model, safe="")
    encoded_key = parse.quote(api_key, safe="")

    if not api_url:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent"
            f"?key={encoded_key}"
        )

    url = api_url
    if "{model}" in url:
        url = url.replace("{model}", encoded_model)
    if "{api_key}" in url:
        url = url.replace("{api_key}", encoded_key)

    if ":generateContent" not in url:
        split = parse.urlsplit(url)
        path = split.path.rstrip("/")
        if path.endswith("/models"):
            new_path = f"{path}/{encoded_model}:generateContent"
        elif "/models/" in path:
            new_path = f"{path}/{encoded_model}:generateContent"
        elif path.endswith("/v1beta"):
            new_path = f"{path}/models/{encoded_model}:generateContent"
        elif not path:
            new_path = f"/v1beta/models/{encoded_model}:generateContent"
        else:
            new_path = f"{path}/v1beta/models/{encoded_model}:generateContent"
        url = parse.urlunsplit((split.scheme, split.netloc, new_path, split.query, split.fragment))

    if "{api_key}" not in api_url and "key=" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}key={encoded_key}"

    return url


def sanitize_url_for_log(url: str) -> str:
    split = parse.urlsplit(url)
    query_pairs = parse.parse_qsl(split.query, keep_blank_values=True)
    safe_pairs = []
    masked_keywords = {"key", "token", "secret", "password", "api_key", "apikey"}

    for key, value in query_pairs:
        lowered = key.lower()
        if any(word in lowered for word in masked_keywords):
            safe_pairs.append((key, "***"))
        else:
            safe_pairs.append((key, value))

    safe_query = parse.urlencode(safe_pairs, doseq=True)
    return parse.urlunsplit((split.scheme, split.netloc, split.path, safe_query, split.fragment))


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    safe_url = sanitize_url_for_log(url)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # 网络中断重连：最多重试 3 次，每次间隔 8 秒
    NETWORK_MAX_RETRIES = 3
    NETWORK_RETRY_INTERVAL = 8  # 秒
    NETWORK_ERRORS = (
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
    )

    http_retries_left = runtime.max_retries
    network_retries_left = NETWORK_MAX_RETRIES if runtime.max_retries > 0 else 0
    network_retries_initial = network_retries_left
    max_total = runtime.max_retries + 1 + NETWORK_MAX_RETRIES

    for _ in range(max_total):
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "pipeline/1.0",
                **headers,
            },
        )
        try:
            with request.urlopen(req, timeout=runtime.request_timeout_seconds) as response:
                text = response.read().decode("utf-8")
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise PipelineError("HTTP 响应不是合法 JSON 对象")
                return parsed
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            retryable = exc.code in {429, 500, 502, 503, 504}
            if retryable and http_retries_left > 0:
                http_retries_left -= 1
                wait = runtime.retry_backoff_seconds * (runtime.max_retries - http_retries_left)
                time.sleep(wait)
                continue
            raise PipelineError(
                f"HTTP 请求失败，URL：{safe_url}，状态码 {exc.code}，响应：{detail[:500]}"
            ) from exc
        except error.URLError as exc:
            if http_retries_left > 0:
                http_retries_left -= 1
                wait = runtime.retry_backoff_seconds * (runtime.max_retries - http_retries_left)
                time.sleep(wait)
                continue
            raise PipelineError(f"网络请求失败，URL：{safe_url}，异常：{exc}") from exc
        except NETWORK_ERRORS as exc:
            if network_retries_left > 0:
                network_retries_left -= 1
                attempt_num = NETWORK_MAX_RETRIES - network_retries_left
                print(
                    f"[network] 连接中断 ({type(exc).__name__}: {exc})，"
                    f"{NETWORK_RETRY_INTERVAL}秒后重试 "
                    f"({attempt_num}/{NETWORK_MAX_RETRIES})..."
                )
                time.sleep(NETWORK_RETRY_INTERVAL)
                continue
            raise PipelineError(
                f"网络连接中断（重试 {network_retries_initial} 次后仍失败），"
                f"URL：{safe_url}，最后异常：{type(exc).__name__}: {exc}"
            ) from exc

    raise PipelineError("HTTP 请求重试后仍失败")


def _parse_openai_sse_stream(response) -> Tuple[str, str]:
    """解析 OpenAI SSE 流式响应，返回 (累积文本, finish_reason)。"""
    accumulated_text: List[str] = []
    finish_reason = ""

    while True:
        raw_line = response.readline()
        if not raw_line:
            break

        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

        if not line or line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    err_msg = err_obj.get("message", str(data))
                else:
                    err_msg = str(err_obj)
                raise PipelineError(f"OpenAI SSE error: {err_msg}")

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    accumulated_text.append(content)

                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr

    # 检查累积文本是否包含中转站嵌入的错误信息
    full_text = "".join(accumulated_text)
    if full_text:
        error_patterns = [
            ("429", "INSUFFICIENT_MODEL_CAPACITY"),
            ("429", "high traffic"),
            ("错误:", "429"),
            ("error:", "429"),
        ]
        text_lower = full_text.lower()
        for p1, p2 in error_patterns:
            if p1.lower() in text_lower and p2.lower() in text_lower:
                raise PipelineError(f"中转站返回嵌入式错误（429 容量不足）：{full_text[:200]}")

    return full_text, finish_reason


def call_openai(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
    api_url: str | None = None,
    extra_messages: List[Dict[str, str]] | None = None,
) -> str:
    url = normalize_openai_url(api_url)
    is_responses_api = parse.urlsplit(url).path.rstrip("/").endswith("/responses")

    if is_responses_api:
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if extra_messages:
            input_messages.extend(extra_messages)
        payload = {
            "model": model,
            "input": input_messages,
            "temperature": 0.2,
        }
        resp = post_json(
            url=url,
            headers={"Authorization": f"Bearer {api_key}"},
            payload=payload,
            runtime=runtime,
        )
        output_text = resp.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = resp.get("output")
        if isinstance(output, list):
            texts: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content_list = item.get("content")
                if not isinstance(content_list, list):
                    continue
                for block in content_list:
                    if not isinstance(block, dict):
                        continue
                    block_type = str(block.get("type", "")).lower()
                    if block_type in {"output_text", "text"}:
                        text = str(block.get("text", "")).strip()
                        if text:
                            texts.append(text)
            merged = "\n".join(texts).strip()
            if merged:
                return merged

        raise PipelineError("OpenAI Responses 响应中未解析到有效文本")

    # Chat Completions API - 使用流式请求
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if extra_messages:
        messages.extend(extra_messages)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }

    return _call_streaming_with_retry(
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
        stream_parser=_parse_openai_sse_stream,
        runtime=runtime,
        provider="openai",
        truncation_signal="length",
    )


def _parse_claude_sse_stream(response) -> Tuple[str, str]:
    """解析 Claude SSE 流式响应，返回 (累积文本, stop_reason)。"""
    accumulated_text: List[str] = []
    stop_reason = ""
    current_event = ""

    while True:
        raw_line = response.readline()
        if not raw_line:
            break

        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue

        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if current_event == "error":
                err_type = data.get("error", {}).get("type", "unknown")
                err_msg = data.get("error", {}).get("message", str(data))
                raise PipelineError(f"Claude SSE error 事件: [{err_type}] {err_msg}")

            if current_event == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        accumulated_text.append(text)

            elif current_event == "message_delta":
                delta = data.get("delta", {})
                sr = delta.get("stop_reason", "")
                if sr:
                    stop_reason = sr

            elif current_event == "message_stop":
                break

        if not line:
            current_event = ""

    return "".join(accumulated_text), stop_reason


_NETWORK_ERRORS = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    ssl.SSLEOFError,
)
_NETWORK_MAX_RETRIES = 3
_NETWORK_RETRY_INTERVAL = 8
_CONTENT_MAX_RETRIES = 3
_CONTENT_RETRY_INTERVAL = 10


def _call_streaming_with_retry(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    stream_parser: Callable,
    runtime: RuntimeConfig,
    provider: str,
    truncation_signal: str,
) -> str:
    """通用流式 LLM 调用 + 重试逻辑。

    provider:          日志标签（openai / claude）
    truncation_signal: 表示截断的 stop_reason 值（length / max_tokens）
    stream_parser:     (response) -> (text, stop_reason)
    """
    safe_url = sanitize_url_for_log(url)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    full_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "pipeline/1.0",
        **headers,
    }

    http_retries_left = runtime.max_retries
    network_retries_left = _NETWORK_MAX_RETRIES if runtime.max_retries > 0 else 0
    network_retries_initial = network_retries_left
    content_retries_left = _CONTENT_MAX_RETRIES if runtime.max_retries > 0 else 0
    content_retries_initial = content_retries_left
    max_total = runtime.max_retries + 1 + network_retries_left + content_retries_left

    for _ in range(max_total):
        req = request.Request(url=url, data=body, method="POST", headers=full_headers)
        try:
            response = request.urlopen(req, timeout=runtime.request_timeout_seconds)
            try:
                accumulated_text, stop_info = stream_parser(response)
            finally:
                response.close()

            if stop_info == truncation_signal:
                print(
                    f"[{provider}] 警告：输出被截断(stop={stop_info})，返回内容可能不完整"
                )

            result = accumulated_text.strip()
            if not result:
                if content_retries_left > 0:
                    content_retries_left -= 1
                    attempt = content_retries_initial - content_retries_left
                    category = classify_error(
                        PipelineError(f"{provider} 流式响应未返回文本")
                    )
                    print(
                        f"[{provider}] 【{category}】流式响应为空，"
                        f"{_CONTENT_RETRY_INTERVAL}秒后重试 "
                        f"({attempt}/{content_retries_initial})..."
                    )
                    time.sleep(_CONTENT_RETRY_INTERVAL)
                    continue
                raise PipelineError(
                    f"{provider} 流式响应未返回文本"
                    f"（重试 {content_retries_initial} 次后仍失败）"
                )

            return result

        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            retryable = exc.code in {429, 500, 502, 503, 504, 524}
            category = classify_error(exc)
            if retryable and http_retries_left > 0:
                http_retries_left -= 1
                wait = runtime.retry_backoff_seconds * (
                    runtime.max_retries - http_retries_left
                )
                attempt = runtime.max_retries - http_retries_left
                print(
                    f"[{provider}] 【{category}】HTTP {exc.code}，{wait}秒后重试 "
                    f"({attempt}/{runtime.max_retries})..."
                )
                time.sleep(wait)
                continue
            raise PipelineError(
                f"HTTP 请求失败，URL：{safe_url}，状态码 {exc.code}，"
                f"错误分类：【{category}】，响应：{detail[:500]}"
            ) from exc

        except error.URLError as exc:
            category = classify_error(exc)
            if http_retries_left > 0:
                http_retries_left -= 1
                wait = runtime.retry_backoff_seconds * (
                    runtime.max_retries - http_retries_left
                )
                attempt = runtime.max_retries - http_retries_left
                print(
                    f"[{provider}] 【{category}】URL 错误 ({exc})，{wait}秒后重试 "
                    f"({attempt}/{runtime.max_retries})..."
                )
                time.sleep(wait)
                continue
            raise PipelineError(
                f"网络请求失败，URL：{safe_url}，错误分类：【{category}】，异常：{exc}"
            ) from exc

        except _NETWORK_ERRORS as exc:
            category = classify_error(exc)
            if network_retries_left > 0:
                network_retries_left -= 1
                attempt = _NETWORK_MAX_RETRIES - network_retries_left
                print(
                    f"[{provider}] 【{category}】连接中断 "
                    f"({type(exc).__name__}: {exc})，"
                    f"{_NETWORK_RETRY_INTERVAL}秒后重试 "
                    f"({attempt}/{network_retries_initial})..."
                )
                time.sleep(_NETWORK_RETRY_INTERVAL)
                continue
            raise PipelineError(
                f"网络连接中断（重试 {network_retries_initial} 次后仍失败），"
                f"URL：{safe_url}，错误分类：【{category}】，"
                f"最后异常：{type(exc).__name__}: {exc}"
            ) from exc

        except PipelineError:
            raise

    raise PipelineError(f"{provider} 流式请求重试后仍失败")


def call_claude(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
    api_url: str | None = None,
    extra_messages: List[Dict[str, str]] | None = None,
) -> str:
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    if extra_messages:
        for em in extra_messages:
            messages.append({
                "role": em["role"],
                "content": em["content"],
            })
    payload = {
        "model": model,
        "system": system_prompt,
        "max_tokens": 32768,
        "stream": True,
        "messages": messages,
    }

    url = normalize_anthropic_url(api_url)
    return _call_streaming_with_retry(
        url=url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        payload=payload,
        stream_parser=_parse_claude_sse_stream,
        runtime=runtime,
        provider="claude",
        truncation_signal="max_tokens",
    )


def call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
    api_url: str | None = None,
    extra_messages: List[Dict[str, str]] | None = None,
) -> str:
    url = normalize_gemini_url(api_url, model, api_key)
    contents = [
        {"role": "user", "parts": [{"text": user_prompt}]},
    ]
    if extra_messages:
        for em in extra_messages:
            role = "model" if em["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": em["content"]}]})
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    resp = post_json(
        url=url,
        headers={},
        payload=payload,
        runtime=runtime,
    )

    candidates = resp.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PipelineError("Gemini 响应中缺少 candidates")

    content = candidates[0].get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    texts: List[str] = []
    for item in parts:
        if isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
    result = "\n".join([text for text in texts if text]).strip()
    if not result:
        raise PipelineError("Gemini 响应未返回文本")
    return result


def call_llm(
    stage: StageConfig,
    endpoint: Endpoint,
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
    extra_messages: List[Dict[str, str]] | None = None,
) -> str:
    # 将 system prompt 合并到 user prompt（兼容不支持 system 字段的中转站）
    if system_prompt.strip():
        merged_user = (
            f"<instructions>\n{system_prompt}\n</instructions>\n\n{user_prompt}"
        )
    else:
        merged_user = user_prompt
    brief_system = "请严格按照 <instructions> 中的指令执行任务。"

    api_type = endpoint.api_type or stage.api_type
    effective_model = endpoint.model or stage.model
    if api_type == "openai":
        return call_openai(
            endpoint.api_key,
            effective_model,
            brief_system,
            merged_user,
            runtime,
            endpoint.api_url,
            extra_messages=extra_messages,
        )
    if api_type == "anthropic":
        return call_claude(
            endpoint.api_key,
            effective_model,
            brief_system,
            merged_user,
            runtime,
            endpoint.api_url,
            extra_messages=extra_messages,
        )
    if api_type == "gemini":
        return call_gemini(
            endpoint.api_key,
            effective_model,
            brief_system,
            merged_user,
            runtime,
            endpoint.api_url,
            extra_messages=extra_messages,
        )
    raise PipelineError(f"不支持的 api_type：{api_type}")


def _create_endpoint_states(
    config: PipelineConfig, stage: StageConfig, log_dir: Path | None = None,
) -> List[EndpointState]:
    """从 StageConfig 创建端点状态列表。支持新旧两种配置格式。"""
    if stage.endpoints:
        return [EndpointState(ep, log_dir=log_dir) for ep in stage.endpoints]
    # 旧格式兼容：api_url + api_key(s)
    api_keys = get_api_keys(config, stage.provider, stage.api_key)
    return [
        EndpointState(Endpoint(
            name=f"key_{i + 1}" if len(api_keys) > 1 else "default",
            api_url=stage.api_url or "",
            api_key=k,
        ), log_dir=log_dir)
        for i, k in enumerate(api_keys)
    ]


def call_llm_with_failover(
    stage: StageConfig,
    endpoint_states: List[EndpointState],
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
    extra_messages: List[Dict[str, str]] | None = None,
) -> str:
    """多端点故障转移调用，带熔断机制。

    故障转移模式下禁用 L1 内部重试（max_retries=0），
    每个端点仅发送 1 次 HTTP 请求，失败立即切换到下一个端点。
    连续失败 3 次的端点触发熔断（冷却 5 小时）。
    """
    available = [es for es in endpoint_states if es.is_available()]
    if not available:
        details = "\n".join(
            f"  - {es.name}: 连续失败 {es.consecutive_failures} 次"
            for es in endpoint_states
        )
        raise EndpointsExhaustedError(
            f"所有端点均已熔断（连续失败 ≥{CIRCUIT_FAILURE_THRESHOLD} 次），"
            f"请检查上游端点配置（模型、API Key、服务状态）后重试。\n{details}"
        )

    # 故障转移模式：禁用 L1 内部重试，每个端点仅 1 次请求机会
    failover_runtime = RuntimeConfig(
        request_timeout_seconds=runtime.request_timeout_seconds,
        max_retries=0,
        retry_backoff_seconds=runtime.retry_backoff_seconds,
    )

    failures: List[str] = []
    for i, es in enumerate(available):
        effective_type = es.endpoint.api_type or stage.api_type
        effective_model = es.endpoint.model or stage.model
        print(f"[failover] 尝试端点 '{es.name}' (model={effective_model}, api_type={effective_type})...")
        try:
            result = call_llm(stage, es.endpoint, system_prompt, user_prompt, failover_runtime, extra_messages)
            es.record_success()
            return result
        except PipelineError as exc:
            es.record_failure(error_msg=str(exc))
            failures.append(f"{es.name}: {exc}")
            print(f"[failover] 端点 '{es.name}' 失败：{exc}")
            next_i = i + 1
            if next_i < len(available):
                print(f"[failover] 切换到 '{available[next_i].name}'")

    raise PipelineError("所有端点均调用失败：" + " | ".join(failures))


# ── 自动续接 ──


_CONTINUATION_MAX_ROUNDS = 5
_CONTINUATION_END_MARKER = "<!-- END -->"

_MERGE_DEDUP_SYSTEM_PROMPT = """\
你是专业文档去重合并工具。
你将收到同一主题下两份已整理的知识文档。请合并为一份，消除重复内容。

## 规则
1. 完全重复的段落只保留一份（选择更完整的版本）
2. 内容实质相同但表述不同的段落，合并为更完整的版本
3. 独特内容必须全部保留
4. 保持 ## 和 ### 层级结构
5. 按逻辑顺序排列（基础概念 → 方法论 → 案例 → 进阶技巧）
6. 保留优先级标注 [P0]/[P1]/[P2]
7. 保留所有具体数据、书名、案例
8. 输出完成后，必须在最后单独一行写 <!-- END --> 作为结束标记"""


def call_llm_with_continuation(
    stage: StageConfig,
    endpoint_states: List[EndpointState],
    system_prompt: str,
    user_prompt: str,
    runtime: RuntimeConfig,
) -> str:
    """带自动续接的 LLM 调用。

    当 system_prompt 中包含结束标记指令时，检测输出是否包含 <!-- END -->，
    若缺失则自动发送「继续」进行多轮续接，最多 _CONTINUATION_MAX_ROUNDS 轮。
    """
    result = call_llm_with_failover(
        stage, endpoint_states, system_prompt, user_prompt, runtime,
    )

    # 如果 system_prompt 未要求结束标记，直接返回（向后兼容）
    if _CONTINUATION_END_MARKER not in system_prompt:
        return result

    # 检查是否完整
    if _CONTINUATION_END_MARKER in result:
        return result.replace(_CONTINUATION_END_MARKER, "").strip()

    # 需要续接
    accumulated_parts = [result]
    extra_messages: List[Dict[str, str]] = []

    for round_num in range(1, _CONTINUATION_MAX_ROUNDS + 1):
        print(
            f"[continuation] 输出未完整（未检测到结束标记），"
            f"第 {round_num}/{_CONTINUATION_MAX_ROUNDS} 轮续接..."
        )
        extra_messages.append({"role": "assistant", "content": result})
        extra_messages.append({"role": "user", "content": "继续"})

        result = call_llm_with_failover(
            stage, endpoint_states, system_prompt, user_prompt, runtime,
            extra_messages=extra_messages,
        )
        accumulated_parts.append(result)

        if _CONTINUATION_END_MARKER in result:
            break

    accumulated = "".join(accumulated_parts)

    if _CONTINUATION_END_MARKER not in accumulated:
        print(
            f"[continuation] 警告：{_CONTINUATION_MAX_ROUNDS} 轮续接后"
            f"仍未检测到结束标记，保存已有内容"
        )

    return accumulated.replace(_CONTINUATION_END_MARKER, "").strip()


# ── 错误分类 ──


def classify_error(exc: BaseException) -> str:
    """根据异常类型/消息/HTTP 状态码返回中文分类标签。"""
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    network_types = (
        "RemoteDisconnected", "IncompleteRead", "ConnectionResetError",
        "ConnectionAbortedError", "BrokenPipeError", "ConnectionRefusedError",
    )
    if exc_type in network_types:
        return "网络中断"

    status_code = None
    if hasattr(exc, "code"):
        status_code = getattr(exc, "code", None)
    if status_code is None:
        import re as _re
        m = _re.search(r"状态码\s*(\d{3})", str(exc))
        if m:
            status_code = int(m.group(1))

    if status_code is not None:
        if status_code in {524, 504, 408}:
            return "HTTP超时"
        if status_code == 429:
            return "请求限流"
        if status_code in {500, 502, 503, 520}:
            return "服务端错误"
        if status_code in {401, 403}:
            return "鉴权失败"

    if "content" in exc_msg and ("null" in exc_msg or "缺少" in exc_msg or "为空" in exc_msg):
        return "响应内容为空"
    if "未返回文本" in str(exc):
        return "响应内容为空"

    if "json" in exc_msg and ("合法" in str(exc) or "解析" in exc_msg or "格式" in exc_msg):
        return "响应格式异常"

    if "max_tokens" in exc_msg or "stop_reason" in exc_msg:
        return "Token超限"

    if "dns" in exc_msg or "name or service not known" in exc_msg:
        return "API地址错误"
    if exc_type == "ConnectionRefusedError" or "连接被拒" in str(exc):
        return "API地址错误"
    if isinstance(exc, error.URLError):
        reason_str = str(getattr(exc, "reason", "")).lower()
        if "name" in reason_str or "resolve" in reason_str or "refused" in reason_str:
            return "API地址错误"

    if any(kw in exc_msg for kw in ("remotedisconnected", "incompleteread", "connection reset",
                                      "connection aborted", "broken pipe")):
        return "网络中断"

    return "未知错误"


def _write_network_retry_log(
    log_dir: Path,
    *,
    phase: str,
    label: str,
    idx: int,
    total: int,
    consecutive: int,
    exc: BaseException,
) -> Path:
    """写入网络重试/连续失败日志。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"network_fail_{timestamp}.log"
    error_category = classify_error(exc)
    lines = [
        f"时间: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"阶段: {phase}",
        f"任务: {label} ({idx}/{total})",
        f"连续失败次数: {consecutive}",
        f"错误分类: 【{error_category}】",
        f"异常类型: {type(exc).__name__}",
        f"异常信息: {exc}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[network] 【{error_category}】失败日志已写入：{log_path}")
    return log_path


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


# ── Markdown 工具函数 ──


def strip_code_fences(text: str) -> str:
    """去掉 LLM 可能包裹的 ```markdown...``` 围栏。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    first_nl = stripped.find("\n")
    if first_nl == -1:
        return stripped

    # 检查首行是否是 fence 开头（```、```markdown 等）
    opening = stripped[:first_nl].strip()
    if not re.match(r"^```(?:markdown|md|text)?$", opening, re.IGNORECASE):
        return stripped

    last_fence = stripped.rfind("```")
    if last_fence <= first_nl:
        return stripped

    return stripped[first_nl + 1:last_fence].strip()


def _compute_extract_fingerprint(extract_dir: Path) -> str:
    """基于 extract 目录所有 .md 文件的内容计算指纹，用于缓存失效检测。"""
    h = hashlib.sha256()
    md_files = sorted(extract_dir.glob("*.md"), key=lambda p: p.name.lower())
    for f in md_files:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _compute_topic_fingerprint(units: List[str]) -> str:
    """基于主题内所有知识单元计算指纹，用于增量缓存。"""
    h = hashlib.sha256()
    for unit in sorted(units):
        h.update(unit.encode("utf-8"))
    return h.hexdigest()[:16]


def _compute_unit_fingerprints(units: List[str]) -> List[str]:
    """计算每条知识单元的独立指纹。"""
    return [hashlib.sha256(u.encode("utf-8")).hexdigest()[:16] for u in units]


def _find_best_cached_match(
    cached_topic_fps: dict[str, dict],
    current_unit_fps: List[str],
    min_overlap: float = 0.5,
) -> tuple[str | None, dict]:
    """基于 unit_fps 集合重叠度查找最佳缓存匹配。

    返回 (cached_topic_name, cached_entry)。
    如果没有任何缓存主题的重叠度 >= min_overlap，返回 (None, {})。
    """
    if not current_unit_fps:
        return None, {}

    current_set = set(current_unit_fps)
    best_name: str | None = None
    best_entry: dict = {}
    best_overlap: float = 0.0

    for name, entry in cached_topic_fps.items():
        if not isinstance(entry, dict):
            continue
        cached_fps = entry.get("unit_fps", [])
        if not cached_fps:
            continue
        cached_set = set(cached_fps)
        intersection = current_set & cached_set
        if not intersection:
            continue
        # 以当前主题和缓存主题中较大的集合为分母，确保双向对称性
        denominator = max(len(current_set), len(cached_set))
        overlap = len(intersection) / denominator
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_entry = entry

    if best_overlap >= min_overlap:
        return best_name, best_entry
    return None, {}


def _save_topic_fingerprints(path: Path, merge_sig: str, fps: dict) -> None:
    """持久化 per-topic 指纹缓存。"""
    dump_json(path, {"merge_sig": merge_sig, "topics": fps})


def _merge_reduce_sub_results(
    sub_results: List[str],
    topic_name: str,
    merge_limit_kb: float,
    stage: "StageConfig",
    endpoint_states: List["EndpointState"],
    runtime: "RuntimeConfig",
    s_tag: str,
    stop_event: threading.Event,
    manifest_lock: threading.Lock,
    consecutive_all_fail: List[int],
    ALL_FAIL_THRESHOLD: int,
) -> str:
    """对子批结果做 pairwise merge-reduce 去重合并。

    每轮将相邻两段合并（两段合计 ≤ merge_limit_kb），直到无法继续。
    合并失败时保留两段原文（降级），EndpointsExhaustedError 则直接上抛。
    """
    items = list(sub_results)
    round_no = 0

    while len(items) > 1:
        if stop_event.is_set():
            break

        next_items: List[str] = []
        merged_any = False
        idx = 0

        while idx < len(items):
            if stop_event.is_set():
                next_items.extend(items[idx:])
                break

            if idx + 1 < len(items):
                a, b = items[idx], items[idx + 1]
                a_kb = len(a.encode("utf-8")) / 1024.0
                b_kb = len(b.encode("utf-8")) / 1024.0

                if a_kb + b_kb <= merge_limit_kb:
                    merge_prompt = (
                        f"以下是关于【{topic_name}】主题的两份已整理文档，请合并去重。\n\n"
                        f"=== 文档 A ===\n{a}\n\n=== 文档 B ===\n{b}"
                    )
                    try:
                        raw = call_llm_with_continuation(
                            stage=stage,
                            endpoint_states=endpoint_states,
                            system_prompt=_MERGE_DEDUP_SYSTEM_PROMPT,
                            user_prompt=merge_prompt,
                            runtime=runtime,
                        )
                        merged = strip_code_fences(raw)
                        next_items.append(merged)
                        merged_any = True
                        with manifest_lock:
                            consecutive_all_fail[0] = 0
                        merged_kb = len(merged.encode("utf-8")) / 1024.0
                        print(
                            f"[synthesize{s_tag}] 合并 round {round_no}: "
                            f"{a_kb:.1f}KB + {b_kb:.1f}KB → {merged_kb:.1f}KB"
                        )
                    except EndpointsExhaustedError:
                        raise
                    except PipelineError as exc:
                        with manifest_lock:
                            consecutive_all_fail[0] += 1
                            if consecutive_all_fail[0] >= ALL_FAIL_THRESHOLD:
                                print(
                                    f"[synthesize{s_tag}] 连续 {consecutive_all_fail[0]} "
                                    f"轮全端点失败，触发熔断终止"
                                )
                                stop_event.set()
                                raise EndpointsExhaustedError(
                                    f"连续 {consecutive_all_fail[0]} 轮全端点调用失败，终止处理"
                                ) from exc
                        print(
                            f"[synthesize{s_tag}] 合并失败 (round {round_no}): {exc}，保留两段原文"
                        )
                        next_items.append(a)
                        next_items.append(b)
                    idx += 2
                else:
                    # 两段合计超限，跳过
                    next_items.append(a)
                    idx += 1
            else:
                # 奇数个，最后一段直接保留
                next_items.append(items[idx])
                idx += 1

        items = next_items
        round_no += 1

        if not merged_any:
            break

    return "\n\n---\n\n".join(items)


def split_large_text(text: str, max_size_kb: float) -> List[str]:
    """按段落边界拆分超大文本为 <= max_size_kb 的块。"""
    max_bytes = int(max_size_kb * 1024)
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    # 按双换行（段落边界）拆分
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current: List[str] = []
    current_size = 0

    for para in paragraphs:
        para_bytes = len(para.encode("utf-8"))

        if current and current_size + para_bytes + 2 > max_bytes:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        # 单段落超限时按行拆分
        if para_bytes > max_bytes:
            lines = para.split("\n")
            for line in lines:
                line_bytes = len(line.encode("utf-8"))
                # 单行超限 → 独立成块并警告
                if line_bytes > max_bytes:
                    print(f"[split] 警告：单行 {line_bytes / 1024:.1f}KB 超过限制 {max_size_kb}KB，独立成块")
                    if current:
                        chunks.append("\n\n".join(current))
                        current = []
                        current_size = 0
                    chunks.append(line)
                    continue
                if current and current_size + line_bytes + 1 > max_bytes:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_size = 0
                current.append(line)
                current_size += line_bytes + 1
        else:
            current.append(para)
            current_size += para_bytes + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


# ── Extract 阶段 ──


def _extract_single(
    config: PipelineConfig,
    endpoint_states: List[EndpointState],
    source_name: str,
    text: str,
) -> str:
    """单次 LLM 调用，返回 Markdown 文本。"""
    user_prompt = f"请处理以下文件。文件名：{source_name}\n\n{text}"
    raw = call_llm_with_continuation(
        stage=config.extract_stage,
        endpoint_states=endpoint_states,
        system_prompt=config.extract_stage.system_prompt,
        user_prompt=user_prompt,
        runtime=config.runtime,
    )
    return strip_code_fences(raw)


def _extract_batch(
    config: PipelineConfig,
    endpoint_states: List[EndpointState],
    files: List[Tuple[str, str]],
) -> str:
    """多文件合批调用。prompt 中用分隔线列出各文件，输出为一整块 Markdown。"""
    parts = []
    for name, text in files:
        parts.append(f"---\n## 来源文件：{name}\n\n{text}")
    user_prompt = "请逐个处理以下文件。\n\n" + "\n\n".join(parts)
    raw = call_llm_with_continuation(
        stage=config.extract_stage,
        endpoint_states=endpoint_states,
        system_prompt=config.extract_stage.system_prompt,
        user_prompt=user_prompt,
        runtime=config.runtime,
    )
    return strip_code_fences(raw)


def get_intermediate_extract_dir(output_dir: Path) -> Path:
    return output_dir / "_intermediate" / "extract"


def _read_manifest(manifest_path: Path) -> set[str]:
    """读取 manifest 文件，返回已处理的源文件名集合。"""
    if not manifest_path.is_file():
        return set()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def _append_manifest(manifest_path: Path, names: List[str]) -> None:
    """追加源文件名到 manifest。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


_TASK_MAX_RETRIES = 3          # L3 层每个任务的最大重试次数
_TASK_RETRY_INTERVAL = 15      # 普通失败后等待秒数


def _extract_with_retry(
    call_fn: Callable[[], str],
    label: str,
    max_retries: int = _TASK_MAX_RETRIES,
) -> str:
    """L3 层重试包装：处理 PipelineError（EndpointsExhaustedError 直接上抛）。"""
    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        try:
            return call_fn()
        except EndpointsExhaustedError:
            # 所有端点已熔断，立即停止，不再重试
            raise
        except PipelineError as exc:
            if attempt > max_retries:
                raise
            print(
                f"[extract] {label}: 失败 ({exc})，"
                f"{_TASK_RETRY_INTERVAL}s 后重试 "
                f"({attempt}/{max_retries})..."
            )
            time.sleep(_TASK_RETRY_INTERVAL)
    # unreachable, but for safety
    raise PipelineError(f"{label}: 重试 {max_retries} 次后仍失败")


# ---------------------------------------------------------------------------
#  并发执行结构化日志
# ---------------------------------------------------------------------------


def _extract_unit_label(unit: Dict[str, Any]) -> str:
    """从 extract 工作单元中提取可读标签。"""
    if unit["type"] == "single":
        return unit["name"]
    elif unit["type"] == "batch":
        return f"批次 {unit['batch_idx']}"
    elif unit["type"] == "chunk_group":
        return unit["name"]
    return str(unit.get("name", "unknown"))


class _ConcurrentLog:
    """线程安全的并发执行日志记录器。

    记录每个工作单元的 worker 名、标签、状态 (ok/error/skip)、耗时和可选错误信息，
    完成后可写入 JSON 文件并打印汇总统计。
    """

    def __init__(self, stage: str, max_workers: int, total_units: int) -> None:
        self.stage = stage
        self.max_workers = max_workers
        self.total_units = total_units
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record(
        self,
        worker: str,
        unit: str,
        status: str,
        duration_s: float,
        error: str | None = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "worker": worker,
            "unit": unit,
            "status": status,
            "duration_s": round(duration_s, 3),
        }
        if error:
            entry["error"] = error[:500]
        with self._lock:
            self._records.append(entry)

    def write_log(self, log_dir: Path) -> Path:
        """将结构化日志写入 JSON 文件，返回文件路径。"""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"concurrent_{self.stage}_{timestamp}.json"
        total_seconds = round(time.time() - self._start_time, 2)

        summary: Dict[str, int] = {"ok": 0, "error": 0, "skip": 0}
        with self._lock:
            for r in self._records:
                summary[r["status"]] = summary.get(r["status"], 0) + 1
            records_snapshot = list(self._records)

        payload = {
            "stage": self.stage,
            "max_workers": self.max_workers,
            "total_units": self.total_units,
            "total_seconds": total_seconds,
            "summary": summary,
            "records": records_snapshot,
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return log_path

    def print_summary(self) -> None:
        """打印汇总统计。"""
        total_seconds = round(time.time() - self._start_time, 2)
        with self._lock:
            ok = sum(1 for r in self._records if r["status"] == "ok")
            err = sum(1 for r in self._records if r["status"] == "error")
            skip = sum(1 for r in self._records if r["status"] == "skip")
        print(
            f"[{self.stage}] 并发统计: ok={ok} error={err} skip={skip} "
            f"总耗时={total_seconds}s workers={self.max_workers}"
        )


def run_extract_stage(
    config: PipelineConfig,
) -> Path:
    """Extract 阶段：.txt → LLM → .md 摘要。返回 extract 目录路径。"""
    source_files = list_source_files(config.input_dir, config.output_dir)
    if not source_files:
        raise PipelineError(f"未在输入目录发现 .txt 文件：{config.input_dir}")

    extract_dir = get_intermediate_extract_dir(config.output_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = extract_dir / "_manifest.txt"

    log_dir = config.output_dir / "_logs"
    endpoint_states = _create_endpoint_states(config, config.extract_stage, log_dir=log_dir)
    batch_cfg = config.batching.extract

    print(f"[extract] 共发现 {len(source_files)} 个 .txt 文件")

    # ── Pass 1: 加载 manifest，识别未处理文件 ──
    done_set = _read_manifest(manifest_path)
    uncached: List[Tuple[Path, str]] = []  # (source_path, source_name)
    for source_path in source_files:
        source_rel = source_path.relative_to(config.input_dir)
        source_name = source_rel.as_posix()
        if source_name in done_set:
            print(f"[extract] {source_name} (已在 manifest 中，跳过)")
            continue
        uncached.append((source_path, source_name))

    if not uncached:
        print(f"[extract] 所有文件均已处理，跳过 API 调用")
        print(f"[extract] 阶段完成，输出目录：{extract_dir}")
        return extract_dir

    # ── Pass 2: 读取未缓存文件，测量大小 ──
    file_data: List[Dict[str, Any]] = []
    for source_path, source_name in uncached:
        text, detected_encoding = read_text_auto(source_path)
        size_kb = len(text.encode("utf-8")) / 1024.0
        file_data.append({
            "source_path": source_path,
            "source_name": source_name,
            "text": text,
            "detected_encoding": detected_encoding,
            "size_kb": size_kb,
        })

    # ── Pass 3: 处理超大文件拆分 + 排序 + 装箱 ──
    # 超大文件阈值：max_batch_size_kb
    effective_limit = get_effective_size_limit(
        batch_cfg.max_batch_size_kb,
        config.extract_stage.provider,
        config.batching.provider_limits,
    )

    # 将文件分为：小文件（可合批）、大文件（单独调用）、超大文件（拆块）
    tasks: List[Dict[str, Any]] = []  # 每项: type=single|batch_item|chunk, ...
    for fd in file_data:
        if fd["size_kb"] >= effective_limit:
            # 超大文件 → 按段落边界拆块
            chunks = split_large_text(fd["text"], effective_limit)
            print(f"[extract] {fd['source_name']} ({fd['size_kb']:.1f}KB) → 拆分为 {len(chunks)} 块")
            for ci, chunk in enumerate(chunks, 1):
                tasks.append({
                    "type": "chunk",
                    "source_name": fd["source_name"],
                    "chunk_index": ci,
                    "chunk_total": len(chunks),
                    "text": chunk,
                    "size_kb": len(chunk.encode("utf-8")) / 1024.0,
                    "detected_encoding": fd["detected_encoding"],
                })
        else:
            tasks.append({
                "type": "normal",
                "source_name": fd["source_name"],
                "text": fd["text"],
                "size_kb": fd["size_kb"],
                "detected_encoding": fd["detected_encoding"],
            })

    # 分离 chunk 任务和 normal 任务
    chunk_tasks = [t for t in tasks if t["type"] == "chunk"]
    normal_tasks = [t for t in tasks if t["type"] == "normal"]

    # Normal 任务按大小排序装箱
    normal_tasks.sort(key=lambda t: t["size_kb"])

    # ── Pass 4: 构建并发工作单元 ──
    work_units: List[Dict[str, Any]] = []

    if batch_cfg.enabled and len(normal_tasks) > 1:
        batch_items = [
            {"source_file": t["source_name"], "content": t["text"]}
            for t in normal_tasks
        ]
        batches = compute_batches(batch_items, batch_cfg.max_files_per_batch, effective_limit)
        normal_map = {t["source_name"]: t for t in normal_tasks}

        print(
            f"[extract] 批处理已启用：{len(normal_tasks)} 个普通文件 → {len(batches)} 个批次, "
            f"{len(chunk_tasks)} 个超大文件拆块"
        )

        for batch_idx, batch in enumerate(batches, start=1):
            batch_names = [item["source_file"] for item in batch]
            if len(batch) == 1:
                name = batch_names[0]
                t = normal_map[name]
                work_units.append({
                    "type": "single",
                    "name": name,
                    "text": t["text"],
                    "encoding": t["detected_encoding"],
                })
            else:
                work_units.append({
                    "type": "batch",
                    "batch_idx": batch_idx,
                    "names": batch_names,
                    "files": [(n, normal_map[n]["text"]) for n in batch_names],
                    "size_kb": sum(normal_map[n]["size_kb"] for n in batch_names),
                })
    else:
        for t in normal_tasks:
            work_units.append({
                "type": "single",
                "name": t["source_name"],
                "text": t["text"],
                "encoding": t["detected_encoding"],
            })

    # 超大文件拆块 → 按源文件分组，同一文件的块在同一线程内顺序处理
    chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for t in chunk_tasks:
        chunks_by_file.setdefault(t["source_name"], []).append(t)
    for cname, clist in chunks_by_file.items():
        clist.sort(key=lambda c: c["chunk_index"])
        work_units.append({
            "type": "chunk_group",
            "name": cname,
            "chunks": clist,
        })

    total_units = len(work_units)

    # ── 并发执行基础设施 ──
    manifest_lock = threading.Lock()
    stop_event = threading.Event()
    progress_lock = threading.Lock()
    done_count = [0]
    worker_id_counter = [0]
    worker_id_lock = threading.Lock()
    clog = _ConcurrentLog("extract", config.concurrency.max_workers, total_units)

    def _process_extract_unit(unit: Dict[str, Any]) -> None:
        """处理单个 extract 工作单元（线程安全）。"""
        if stop_event.is_set():
            clog.record(
                worker=threading.current_thread().name,
                unit=_extract_unit_label(unit),
                status="skip",
                duration_s=0.0,
            )
            return

        with progress_lock:
            done_count[0] += 1
            seq = done_count[0]

        # 为并发模式分配 worker ID
        w_tag = ""
        if max_workers > 1:
            with worker_id_lock:
                wid = worker_id_counter[0]
                worker_id_counter[0] += 1
            w_tag = f" W{wid % max_workers}"

        t0 = time.time()
        unit_label = _extract_unit_label(unit)
        worker_name = threading.current_thread().name

        try:
            if unit["type"] == "single":
                name = unit["name"]
                print(f"[extract{w_tag} {seq}/{total_units}] {name} (编码: {unit['encoding']})")
                md = _extract_with_retry(
                    lambda _n=name, _t=unit["text"]: _extract_single(config, endpoint_states, _n, _t),
                    label=name,
                )
                out_path = extract_dir / f"{safe_filename(name)}.md"
                out_path.write_text(md, encoding="utf-8")
                with manifest_lock:
                    _append_manifest(manifest_path, [name])
                print(f"[extract{w_tag}] → {out_path.name}")

            elif unit["type"] == "batch":
                batch_idx = unit["batch_idx"]
                batch_names = unit["names"]
                files_for_prompt = unit["files"]
                print(
                    f"[extract{w_tag} {seq}/{total_units}] 批次 {batch_idx}: "
                    f"{len(batch_names)} 个文件, {unit['size_kb']:.1f}KB"
                )
                try:
                    md = _extract_with_retry(
                        lambda _f=files_for_prompt: _extract_batch(config, endpoint_states, _f),
                        label=f"批次 {batch_idx}",
                    )
                    out_path = extract_dir / _batch_output_name(batch_names)
                    out_path.write_text(md, encoding="utf-8")
                    with manifest_lock:
                        _append_manifest(manifest_path, batch_names)
                    print(f"[extract{w_tag}] → {out_path.name}")
                except EndpointsExhaustedError:
                    raise
                except PipelineError as exc:
                    # 批量失败 → 逐文件回退
                    print(f"[extract{w_tag}] 批次失败（重试耗尽）: {exc}，逐文件回退")
                    for fi, (fname, ftext) in enumerate(files_for_prompt, 1):
                        if stop_event.is_set():
                            clog.record(worker=worker_name, unit=unit_label, status="skip", duration_s=time.time() - t0)
                            return
                        print(f"[extract{w_tag}] 回退处理: {fname} ({fi}/{len(files_for_prompt)})")
                        try:
                            md = _extract_with_retry(
                                lambda _n=fname, _t=ftext: _extract_single(config, endpoint_states, _n, _t),
                                label=fname,
                            )
                            out_path = extract_dir / f"{safe_filename(fname)}.md"
                            out_path.write_text(md, encoding="utf-8")
                            with manifest_lock:
                                _append_manifest(manifest_path, [fname])
                        except EndpointsExhaustedError:
                            raise
                        except PipelineError as single_exc:
                            print(f"[extract{w_tag}] 单文件也失败（重试耗尽）: {fname}: {single_exc}")

            elif unit["type"] == "chunk_group":
                name = unit["name"]
                chunks = unit["chunks"]
                for t in chunks:
                    if stop_event.is_set():
                        clog.record(worker=worker_name, unit=unit_label, status="skip", duration_s=time.time() - t0)
                        return
                    ci = t["chunk_index"]
                    ct = t["chunk_total"]
                    print(f"[extract{w_tag} {seq}/{total_units}] {name} 块 {ci}/{ct}")
                    chunk_label = f"{name} (块 {ci}/{ct})"
                    md = _extract_with_retry(
                        lambda _l=chunk_label, _t=t["text"]: _extract_single(config, endpoint_states, _l, _t),
                        label=chunk_label,
                    )
                    out_path = extract_dir / f"{safe_filename(name)}_part{ci}.md"
                    out_path.write_text(md, encoding="utf-8")
                # 所有块处理完后才写 manifest
                with manifest_lock:
                    _append_manifest(manifest_path, [name])

            clog.record(worker=worker_name, unit=unit_label, status="ok", duration_s=time.time() - t0)

        except EndpointsExhaustedError:
            clog.record(worker=worker_name, unit=unit_label, status="error", duration_s=time.time() - t0, error="EndpointsExhausted")
            stop_event.set()
            raise
        except PipelineError as exc:
            clog.record(worker=worker_name, unit=unit_label, status="error", duration_s=time.time() - t0, error=str(exc))
            print(f"[extract{w_tag}] 失败（重试耗尽）: {exc}")

    # ── 执行（串行 or 并发） ──
    max_workers = config.concurrency.max_workers
    # 工作单元数少于 max_workers 时强制串行，避免少量任务下线程争用端点锁
    use_serial = max_workers <= 1 or len(work_units) < max_workers

    if use_serial:
        if max_workers > 1 and len(work_units) < max_workers:
            print(f"[extract] 工作单元({len(work_units)})少于并发数({max_workers})，使用串行模式")
        for unit in work_units:
            _process_extract_unit(unit)
            if stop_event.is_set():
                break
    else:
        print(f"[extract] 并发模式：max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="extract") as executor:
            futures = {executor.submit(_process_extract_unit, u): u for u in work_units}
            try:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        for pending_f in futures:
                            pending_f.cancel()
                        break
                    try:
                        future.result()
                    except EndpointsExhaustedError:
                        for pending_f in futures:
                            pending_f.cancel()
                        break
            except KeyboardInterrupt:
                print("\n[extract] 用户中断，取消剩余任务...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    # ── 并发日志输出 ──
    if not use_serial:
        clog.write_log(log_dir)
        clog.print_summary()

    print(f"[extract] 阶段完成，输出目录：{extract_dir}")
    return extract_dir


# ── Synthesize 阶段 ──


def split_topics_markdown(text: str) -> List[Tuple[str, str]]:
    """按一级标题 (^# ) 拆分为 [(filename, content), ...]"""
    chunks = re.split(r"\n(?=# )", text.lstrip("\n"))
    topics: List[Tuple[str, str]] = []
    for i, chunk in enumerate(chunks, 1):
        chunk = chunk.strip()
        if not chunk:
            continue
        title = re.sub(r"^#+\s*", "", chunk.split("\n", 1)[0]).strip()
        filename = f"{i:02d}_{safe_filename(title)}.md"
        topics.append((filename, chunk))
    return topics


def load_extract_results_from_disk(extract_dir: Path) -> str:
    """读取 extract 目录下所有 .md 文件（排除 _manifest.txt），拼接返回。"""
    if not extract_dir.is_dir():
        raise PipelineError(f"extract 中间目录不存在：{extract_dir}")

    md_files = sorted(
        [f for f in extract_dir.glob("*.md")],
        key=lambda p: p.name.lower(),
    )
    if not md_files:
        raise PipelineError(f"extract 中间目录中无 .md 文件：{extract_dir}")

    parts: List[str] = []
    for f in md_files:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            parts.append(content)

    if not parts:
        raise PipelineError("未读取到有效的 extract 结果")

    return "\n\n---\n\n".join(parts)


# ── 知识单元解析与分类工具 ──


def parse_knowledge_units(all_extracts_text: str) -> List[Dict[str, str]]:
    """将 extract 合并文本按 ## 标题拆分为独立知识单元。

    只识别二级标题(##)作为知识单元边界。
    一级标题(#)视为文件分组标记，跳过。
    三级及以下标题(###/####)包含在所属知识单元内。
    """
    units: List[Dict[str, str]] = []
    current_title: str | None = None
    current_lines: List[str] = []

    for line in all_extracts_text.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            if current_title is not None:
                units.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                })
            current_title = line[3:].strip()
            current_lines = [line]
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        units.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
        })

    return units


def _extract_priority(title: str) -> int:
    """从标题中提取优先级。P0=0, P1=1, P2=2, 无标注=3。"""
    if "[P0]" in title:
        return 0
    if "[P1]" in title:
        return 1
    if "[P2]" in title:
        return 2
    return 3


def _repair_json(text: str) -> Any:
    """修复 LLM 输出中常见的 JSON 格式问题。"""
    text = strip_code_fences(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise PipelineError(f"无法解析 JSON：{text[:300]}")


def group_by_category(extract_dir: Path) -> tuple[
    Dict[str, List[KnowledgeUnit]],  # tag → units
    List[KnowledgeUnit],             # 全量 units 列表
]:
    """读取 extract 目录所有 .md，按 ## [分类标签][Px] 标签归组为结构化知识单元。

    返回 (tag_buckets, all_units)。
    未匹配标签格式的 ## 单元归入 "未分类"。
    """
    if not extract_dir.is_dir():
        raise PipelineError(f"extract 目录不存在：{extract_dir}")

    md_files = sorted(
        [f for f in extract_dir.glob("*.md")],
        key=lambda p: p.name.lower(),
    )
    if not md_files:
        raise PipelineError(f"extract 目录中无 .md 文件：{extract_dir}")

    # 同时捕获标签、优先级、标题
    tag_re = re.compile(r'^##\s*\[([^\]]+)\]\s*\[P([012])\]\s*(.*)')

    buckets: Dict[str, List[KnowledgeUnit]] = {}
    all_units: List[KnowledgeUnit] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        # 按 ## (非 ###) 边界拆分知识单元
        units = re.split(r'\n(?=## (?!#))', text)

        for unit_text in units:
            unit_text = unit_text.strip()
            if not unit_text or not unit_text.startswith("## "):
                continue

            first_line = unit_text.split("\n", 1)[0]
            match = tag_re.match(first_line)
            if match:
                tag = match.group(1).strip()
                priority = int(match.group(2))
                title = match.group(3).strip()
            else:
                tag = "未分类"
                priority = _extract_priority(first_line)
                title = re.sub(r'^##\s*(\[[^\]]*\]\s*)*', '', first_line).strip()

            fp = hashlib.sha256(unit_text.encode("utf-8")).hexdigest()[:16]
            ku = KnowledgeUnit(
                tag=tag, priority=priority, title=title,
                body=unit_text, fingerprint=fp,
            )
            buckets.setdefault(tag, []).append(ku)
            all_units.append(ku)

    return buckets, all_units


def apply_topic_mapping(
    tag_buckets: Dict[str, List[KnowledgeUnit]],
    mapping: Dict[str, List[str]],
) -> tuple[Dict[str, List[KnowledgeUnit]], List[str]]:
    """按 mapping 将标签归入主题，未映射标签归入'未分类'。

    返回 (topic_buckets, unmapped_tags)。
    每个主题内按优先级排序：P0 → P1 → P2。
    """
    tag_to_topic: Dict[str, str] = {}
    for topic_name, tags in mapping.items():
        for tag in tags:
            tag = tag.strip()
            if tag and tag not in tag_to_topic:
                tag_to_topic[tag] = topic_name

    topic_buckets: Dict[str, List[KnowledgeUnit]] = {}
    unmapped_tags: List[str] = []

    for tag_name, units in tag_buckets.items():
        topic = tag_to_topic.get(tag_name)
        if topic is None:
            unmapped_tags.append(tag_name)
            topic = "未分类"
        topic_buckets.setdefault(topic, []).extend(units)

    # 每个主题内按优先级排序
    for topic_name in topic_buckets:
        topic_buckets[topic_name].sort(key=lambda u: u.priority)

    return topic_buckets, unmapped_tags


# ── Embedding 向量语义去重 ──


def _embedding_url_from_chat_url(chat_url: str) -> str:
    """将 /chat/completions 端点 URL 转换为 /embeddings 端点 URL。"""
    url = chat_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if url.endswith(suffix):
            return url[: -len(suffix)] + "/embeddings"
    # 如果 URL 不是标准格式，直接追加 /embeddings
    return url + "/embeddings"


def _get_embeddings(
    texts: List[str],
    config: PipelineConfig,
    endpoint_states: List["EndpointState"],
    sim_cfg: "SimilarityConfig",
) -> List[List[float]]:
    """批量获取文本 embedding 向量。

    优先使用 sim_cfg 中配置的独立 embedding 端点（embedding_api_url）；
    若未配置则 fallback 到 synthesize 端点（将 /chat/completions 替换为 /embeddings）。
    返回与输入同序的向量列表。
    """
    model = sim_cfg.embedding_model
    batch_size = sim_cfg.embedding_batch_size
    use_dedicated = bool(sim_cfg.embedding_api_url)

    all_vectors: List[List[float]] = [[] for _ in range(len(texts))]

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_texts)))

        payload = {
            "model": model,
            "input": batch_texts,
        }

        def _parse_embedding_response(resp: dict) -> List[List[float]]:
            data = resp.get("data", [])
            if not isinstance(data, list) or len(data) != len(batch_texts):
                raise PipelineError(
                    f"Embedding 响应格式异常: 期望 {len(batch_texts)} 条, "
                    f"实际 {len(data) if isinstance(data, list) else 'N/A'}"
                )
            sorted_data = sorted(data, key=lambda d: d.get("index", 0))
            return [item.get("embedding", []) for item in sorted_data]

        if use_dedicated:
            # 独立 embedding 端点：直接调用，不走 failover
            try:
                resp = post_json(
                    url=sim_cfg.embedding_api_url,
                    headers={"Authorization": f"Bearer {sim_cfg.embedding_api_key}"},
                    payload=payload,
                    runtime=config.runtime,
                )
                vectors = _parse_embedding_response(resp)
                for i, vec in enumerate(vectors):
                    all_vectors[batch_indices[i]] = vec
            except PipelineError as exc:
                raise PipelineError(
                    f"Embedding 批次 {batch_start}-{batch_start + len(batch_texts)} "
                    f"独立端点失败: {exc}"
                ) from exc
        else:
            # Fallback：遍历 synthesize 端点
            available = [es for es in endpoint_states if es.is_available()]
            if not available:
                raise EndpointsExhaustedError("所有端点均已熔断，无法获取 embedding")

            success = False
            for es in available:
                ep = es.endpoint
                embedding_url = _embedding_url_from_chat_url(ep.api_url)
                payload["model"] = ep.model if ep.model else model
                try:
                    resp = post_json(
                        url=embedding_url,
                        headers={"Authorization": f"Bearer {ep.api_key}"},
                        payload=payload,
                        runtime=config.runtime,
                    )
                    es.record_success()
                    vectors = _parse_embedding_response(resp)
                    for i, vec in enumerate(vectors):
                        all_vectors[batch_indices[i]] = vec
                    success = True
                    break
                except (PipelineError, EndpointsExhaustedError) as exc:
                    es.record_failure(error_msg=str(exc))
                    print(f"[embedding] 端点 '{es.name}' 失败: {exc}")
                    continue

            if not success:
                raise PipelineError(
                    f"Embedding 批次 {batch_start}-{batch_start + len(batch_texts)} 所有端点均失败"
                )

        if batch_start + batch_size < len(texts):
            done = min(batch_start + batch_size, len(texts))
            print(f"[embedding] 进度: {done}/{len(texts)}")

    return all_vectors


def _load_embedding_cache(cache_path: Path) -> Dict[str, List[float]]:
    """加载 embedding 缓存。key = unit fingerprint, value = vector。"""
    if not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_embedding_cache(cache_path: Path, cache: Dict[str, List[float]]) -> None:
    """保存 embedding 缓存（原子写入：写临时文件再 rename）。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    text = json.dumps(cache, ensure_ascii=False)
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(cache_path)


def _cosine_matrix(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """计算两组向量的余弦相似度矩阵。

    输入: vecs_a (M, D), vecs_b (N, D)
    输出: (M, N) 余弦相似度矩阵
    """
    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    # 避免除零
    norms_a = np.maximum(norms_a, 1e-10)
    norms_b = np.maximum(norms_b, 1e-10)
    a_norm = vecs_a / norms_a
    b_norm = vecs_b / norms_b
    return a_norm @ b_norm.T


def scan_similarity_embedding(
    topic_units: Dict[str, List[KnowledgeUnit]],
    all_units: List[KnowledgeUnit],
    cached_fps: set,
    threshold: float,
    embeddings: Dict[str, List[float]],
) -> tuple[List[tuple[KnowledgeUnit, KnowledgeUnit, float]],
           List[tuple[KnowledgeUnit, KnowledgeUnit, float]]]:
    """Embedding 余弦相似度扫描。

    返回值与 scan_similarity() 相同:
    - intra_pairs: 同主题内超阈值的对（送 LLM）
    - cross_pairs: 跨主题超阈值的对（仅报告）
    """
    intra_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []
    cross_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []

    # 识别新增单元
    new_fps = {u.fingerprint for u in all_units} - cached_fps

    # 同主题内扫描：新↔新 + 新↔旧
    total_intra_comparisons = 0
    for topic_name, units in topic_units.items():
        # 过滤掉没有 embedding 的单元
        units_with_emb = [u for u in units if u.fingerprint in embeddings]
        new_in_topic = [u for u in units_with_emb if u.fingerprint in new_fps]
        old_in_topic = [u for u in units_with_emb if u.fingerprint not in new_fps]

        if not new_in_topic:
            continue

        # 新↔新 扫描
        if len(new_in_topic) >= 2:
            vecs = np.array([embeddings[u.fingerprint] for u in new_in_topic], dtype=np.float32)
            sim_matrix = _cosine_matrix(vecs, vecs)
            n = len(new_in_topic)
            total_intra_comparisons += n * (n - 1) // 2
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sim_matrix[i, j])
                    if sim >= threshold:
                        intra_pairs.append((new_in_topic[i], new_in_topic[j], sim))

        # 新↔旧 扫描
        if new_in_topic and old_in_topic:
            new_vecs = np.array([embeddings[u.fingerprint] for u in new_in_topic], dtype=np.float32)
            old_vecs = np.array([embeddings[u.fingerprint] for u in old_in_topic], dtype=np.float32)
            sim_matrix = _cosine_matrix(new_vecs, old_vecs)
            total_intra_comparisons += len(new_in_topic) * len(old_in_topic)
            for i in range(len(new_in_topic)):
                for j in range(len(old_in_topic)):
                    sim = float(sim_matrix[i, j])
                    if sim >= threshold:
                        intra_pairs.append((new_in_topic[i], old_in_topic[j], sim))

        if new_in_topic or old_in_topic:
            pairs_checked = (
                len(new_in_topic) * (len(new_in_topic) - 1) // 2
                + len(new_in_topic) * len(old_in_topic)
            )
            if pairs_checked > 0:
                print(f"[synthesize] 扫描 {topic_name}: {len(new_in_topic)} 新 × {len(old_in_topic)} 旧 = {pairs_checked} 对")

    print(f"[synthesize] 同主题扫描完成: {total_intra_comparisons} 次比较, {len(intra_pairs)} 对超过阈值")

    # 跨主题扫描：仅对新单元采样（限制计算量）
    CROSS_TOPIC_SAMPLE = 50
    topic_names = list(topic_units.keys())
    cross_comparisons = 0

    for ti in range(len(topic_names)):
        for tj in range(ti + 1, len(topic_names)):
            units_a = topic_units[topic_names[ti]]
            units_b = topic_units[topic_names[tj]]
            new_a = [u for u in units_a if u.fingerprint in new_fps and u.fingerprint in embeddings][:CROSS_TOPIC_SAMPLE]
            new_b = [u for u in units_b if u.fingerprint in new_fps and u.fingerprint in embeddings][:CROSS_TOPIC_SAMPLE]

            if not new_a or not new_b:
                continue

            vecs_a = np.array([embeddings[u.fingerprint] for u in new_a], dtype=np.float32)
            vecs_b = np.array([embeddings[u.fingerprint] for u in new_b], dtype=np.float32)
            sim_matrix = _cosine_matrix(vecs_a, vecs_b)
            cross_comparisons += len(new_a) * len(new_b)

            for i in range(len(new_a)):
                for j in range(len(new_b)):
                    sim = float(sim_matrix[i, j])
                    if sim >= threshold:
                        cross_pairs.append((new_a[i], new_b[j], sim))

    print(f"[synthesize] 跨主题扫描完成: {cross_comparisons} 次比较, {len(cross_pairs)} 对超过阈值")

    return intra_pairs, cross_pairs


def ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """计算两段文本的字符级 n-gram Jaccard 相似度。"""
    def ngrams(text: str, n: int) -> set:
        cleaned = re.sub(r'[#*\-|>\[\](){}]', '', text)
        cleaned = re.sub(r'\s+', '', cleaned)
        if len(cleaned) < n:
            return set()
        return set(cleaned[i:i+n] for i in range(len(cleaned) - n + 1))
    a, b = ngrams(text_a, n), ngrams(text_b, n)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_ngrams(text: str, n: int) -> frozenset:
    """预计算文本的字符级 n-gram 集合。"""
    cleaned = re.sub(r'[#*\-|>\[\](){}]', '', text)
    cleaned = re.sub(r'\s+', '', cleaned)
    if len(cleaned) < n:
        return frozenset()
    return frozenset(cleaned[i:i+n] for i in range(len(cleaned) - n + 1))


def _jaccard_from_sets(a: frozenset, b: frozenset) -> float:
    """从预计算的 n-gram 集合计算 Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def scan_similarity(
    topic_units: Dict[str, List[KnowledgeUnit]],
    all_units: List[KnowledgeUnit],
    cached_fps: set,
    threshold: float,
    ngram_size: int,
) -> tuple[List[tuple[KnowledgeUnit, KnowledgeUnit, float]], List[tuple[KnowledgeUnit, KnowledgeUnit, float]]]:
    """增量相似度扫描（优化版：预计算 n-gram）。

    返回 (intra_topic_pairs, cross_topic_pairs)。
    intra: 同主题内 ≥ threshold 的对（送 LLM）
    cross: 跨主题 ≥ threshold 的对（仅报告）
    """
    intra_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []
    cross_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []

    # 建立 unit → topic 索引
    unit_topic: Dict[str, str] = {}
    for topic_name, units in topic_units.items():
        for u in units:
            unit_topic[u.fingerprint] = topic_name

    # 识别新增单元
    new_fps = {u.fingerprint for u in all_units} - cached_fps

    # 预计算所有单元的 n-gram 集合（一次性）
    print(f"[synthesize] 预计算 {len(all_units)} 条单元的 n-gram...")
    ngram_cache: Dict[str, frozenset] = {}
    for u in all_units:
        if u.fingerprint not in ngram_cache:
            ngram_cache[u.fingerprint] = _compute_ngrams(u.body, ngram_size)

    # 同主题内扫描：新↔新 + 新↔旧
    total_intra_comparisons = 0
    for topic_name, units in topic_units.items():
        new_in_topic = [u for u in units if u.fingerprint in new_fps]
        old_in_topic = [u for u in units if u.fingerprint not in new_fps]

        pairs_to_check = (
            len(new_in_topic) * (len(new_in_topic) - 1) // 2
            + len(new_in_topic) * len(old_in_topic)
        )
        if pairs_to_check == 0:
            continue

        total_intra_comparisons += pairs_to_check
        print(f"[synthesize] 扫描 {topic_name}: {len(new_in_topic)} 新 × {len(old_in_topic)} 旧 = {pairs_to_check} 对")

        # 新↔新
        for i in range(len(new_in_topic)):
            ng_i = ngram_cache[new_in_topic[i].fingerprint]
            for j in range(i + 1, len(new_in_topic)):
                sim = _jaccard_from_sets(ng_i, ngram_cache[new_in_topic[j].fingerprint])
                if sim >= threshold:
                    intra_pairs.append((new_in_topic[i], new_in_topic[j], sim))

        # 新↔旧
        for new_u in new_in_topic:
            ng_new = ngram_cache[new_u.fingerprint]
            for old_u in old_in_topic:
                sim = _jaccard_from_sets(ng_new, ngram_cache[old_u.fingerprint])
                if sim >= threshold:
                    intra_pairs.append((new_u, old_u, sim))

    print(f"[synthesize] 同主题扫描完成: {total_intra_comparisons} 次比较, {len(intra_pairs)} 对超过阈值")

    # 跨主题扫描：仅对新单元采样（限制计算量）
    # 每个主题最多取 50 条新单元做跨主题扫描
    CROSS_TOPIC_SAMPLE = 50
    topic_names = list(topic_units.keys())
    cross_comparisons = 0

    for ti in range(len(topic_names)):
        for tj in range(ti + 1, len(topic_names)):
            units_a = topic_units[topic_names[ti]]
            units_b = topic_units[topic_names[tj]]
            new_a = [u for u in units_a if u.fingerprint in new_fps][:CROSS_TOPIC_SAMPLE]
            new_b = [u for u in units_b if u.fingerprint in new_fps][:CROSS_TOPIC_SAMPLE]

            # 双向：A的新↔B的新
            for ua in new_a:
                ng_a = ngram_cache[ua.fingerprint]
                for ub in new_b:
                    sim = _jaccard_from_sets(ng_a, ngram_cache[ub.fingerprint])
                    cross_comparisons += 1
                    if sim >= threshold:
                        cross_pairs.append((ua, ub, sim))

    print(f"[synthesize] 跨主题扫描完成: {cross_comparisons} 次比较, {len(cross_pairs)} 对超过阈值")

    return intra_pairs, cross_pairs


def build_dedup_prompt(pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]]) -> str:
    """构建批量 verdict 判断 prompt（仅判定，不合并）。"""
    lines = [
        "请判断以下知识单元对是否重复。\n"
        "- DUPLICATE：完全重复，指出保留 A 还是 B（优先保留 P0）\n"
        "- MERGE：部分重复有互补内容，需要后续合并\n"
        "- KEEP_BOTH：不重复，各有独特价值\n"
        "\n请严格按以下格式输出，每对一行，不要输出其他内容：\n"
        "对号|判定|保留侧\n"
        "示例：\n"
        "1|KEEP_BOTH\n"
        "2|DUPLICATE|A\n"
        "3|MERGE\n"
    ]
    for idx, (a, b, sim) in enumerate(pairs, 1):
        lines.append(f"\n【对{idx}】（相似度: {sim:.0%}）")
        lines.append(f"A (P{a.priority}): [{a.tag}] {a.title}")
        body_a = a.body[:2000] + ("..." if len(a.body) > 2000 else "")
        body_b = b.body[:2000] + ("..." if len(b.body) > 2000 else "")
        lines.append(body_a)
        lines.append(f"\nB (P{b.priority}): [{b.tag}] {b.title}")
        lines.append(body_b)

    return "\n".join(lines)


def _parse_verdicts(
    raw: str, num_pairs: int,
) -> List[dict]:
    """解析 verdict 批量输出。

    期望格式：每行 '对号|判定' 或 '对号|判定|保留侧'。
    兼容各种分隔符（| : 空格 制表符）和 LLM 可能附加的文字。
    返回列表，索引 = pair_idx (0-based)，值 = {"verdict": ..., "keep": ...}。
    """
    results: List[dict] = [{"verdict": "KEEP_BOTH"} for _ in range(num_pairs)]
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        # 统一分隔符
        normalized = re.sub(r'[\|\:\t]+', '|', line)
        parts = [p.strip() for p in normalized.split('|') if p.strip()]
        if len(parts) < 2:
            continue
        try:
            pair_num = int(re.match(r'\d+', parts[0]).group())  # type: ignore[union-attr]
        except (ValueError, AttributeError):
            continue
        idx = pair_num - 1
        if not (0 <= idx < num_pairs):
            continue
        verdict = parts[1].upper()
        if verdict not in ("DUPLICATE", "MERGE", "KEEP_BOTH"):
            # 容错：包含关键词即可
            if "DUPLICATE" in verdict or "DUP" in verdict:
                verdict = "DUPLICATE"
            elif "MERGE" in verdict:
                verdict = "MERGE"
            else:
                verdict = "KEEP_BOTH"
        entry: dict = {"verdict": verdict}
        if verdict == "DUPLICATE" and len(parts) >= 3:
            keep_side = parts[2].upper().strip()
            if keep_side in ("A", "B"):
                entry["keep"] = keep_side
        results[idx] = entry
    return results


def build_merge_prompt(a: KnowledgeUnit, b: KnowledgeUnit, sim: float) -> str:
    """构建单对合并 prompt，要求 LLM 返回合并后的 Markdown 文本。"""
    return (
        "请将以下两条知识单元合并为一条，保留双方独特内容，消除重复部分。\n"
        "输出要求：直接输出合并后的完整 Markdown 文本（以 ## 开头），不要加任何包裹或说明。\n"
        f"输出完成后，必须在最后单独一行写 {_CONTINUATION_END_MARKER} 作为结束标记。\n"
        f"\n【A】(P{a.priority}): [{a.tag}] {a.title}\n"
        f"{a.body}\n"
        f"\n【B】(P{b.priority}): [{b.tag}] {b.title}\n"
        f"{b.body}\n"
    )


def apply_dedup_verdicts(
    topic_units: Dict[str, List[KnowledgeUnit]],
    judgments: Dict[str, dict],
) -> Dict[str, List[KnowledgeUnit]]:
    """根据判断结果去重/合并，返回去重后的 topic → units。"""
    # 收集需要删除的指纹和需要替换的指纹
    drop_fps: set = set()
    replacements: Dict[str, KnowledgeUnit] = {}  # fp → replacement unit
    merge_involved_fps: set = set()  # 参与 MERGE 的所有 fp

    for pair_key, judgment in judgments.items():
        verdict = judgment.get("verdict", "KEEP_BOTH")
        if verdict == "MERGE":
            merged_text = judgment.get("merged_text", "")
            if merged_text:
                fps = pair_key.split("_")
                if len(fps) == 2:
                    merge_involved_fps.update(fps)
                    # 找到两条原始单元，取优先级更高（数值更小）的元信息
                    candidates: List[KnowledgeUnit] = []
                    for units in topic_units.values():
                        for u in units:
                            if u.fingerprint in fps:
                                candidates.append(u)
                    if candidates:
                        best = min(candidates, key=lambda x: x.priority)
                        new_fp = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()[:16]
                        keep_fp = fps[0]
                        drop_fps.add(fps[1])
                        replacements[keep_fp] = KnowledgeUnit(
                            tag=best.tag,
                            priority=best.priority,
                            title=best.title,
                            body=merged_text,
                            fingerprint=new_fp,
                        )
        elif verdict == "DUPLICATE":
            drop_fp = judgment.get("drop", "")
            if drop_fp:
                # MERGE 优先：如果该 fp 已被 MERGE 处理，则忽略 DUPLICATE
                if drop_fp not in merge_involved_fps:
                    drop_fps.add(drop_fp)

    result: Dict[str, List[KnowledgeUnit]] = {}
    for topic_name, units in topic_units.items():
        filtered: List[KnowledgeUnit] = []
        for u in units:
            if u.fingerprint in drop_fps:
                continue
            if u.fingerprint in replacements:
                filtered.append(replacements[u.fingerprint])
            else:
                filtered.append(u)
        # 按优先级排序
        filtered.sort(key=lambda x: x.priority)
        result[topic_name] = filtered
    return result


def _atomic_write(path: Path, content: str) -> None:
    """原子写入：先写临时文件，再 os.replace 到目标路径。"""
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_topic_files(
    topic_units: Dict[str, List[KnowledgeUnit]],
    topics_dir: Path,
    max_kb: int,
    topic_fp_cache: dict,
) -> Tuple[List[str], Dict[str, str]]:
    """按优先级排序 + 100KB 分卷写入主题文件。

    返回 (写入的文件名列表, unit fingerprint→filename 映射)。
    """
    written_files: List[str] = []
    unit_file_map: Dict[str, str] = {}  # fingerprint → filename

    for topic_name, units in sorted(topic_units.items(), key=lambda kv: (kv[0] == "未分类", kv[0])):
        if not units:
            continue

        safe_name = safe_filename(topic_name)

        # 计算内容指纹
        content_fp = hashlib.sha256(
            "".join(u.body for u in units).encode("utf-8")
        ).hexdigest()[:16]

        # 检查缓存
        cached = topic_fp_cache.get(topic_name, {})
        if isinstance(cached, dict) and cached.get("fingerprint") == content_fp:
            # 检查文件是否存在
            cached_files = cached.get("files", [])
            if cached_files and all((topics_dir / f).is_file() for f in cached_files):
                print(f"[synthesize] {topic_name}: 内容未变，跳过写入")
                written_files.extend(cached_files)
                continue

        # 拼接全部内容
        full_text = "\n\n".join(u.body for u in units)
        full_bytes = len(full_text.encode("utf-8"))
        max_bytes = max_kb * 1024

        # 清理该主题的旧文件（精确匹配：base.md + base-NNN.md）
        base_name = f"{safe_name}.md"
        for pattern in [f"{safe_name}.md", f"{safe_name}-*.md"]:
            for old_file in topics_dir.glob(pattern):
                old_file.unlink()

        if full_bytes <= max_bytes:
            # 单文件
            file_path = topics_dir / base_name
            _atomic_write(file_path, full_text)
            written_files.append(base_name)
            for u in units:
                unit_file_map[u.fingerprint] = base_name
            topic_fp_cache[topic_name] = {"fingerprint": content_fp, "files": [base_name]}
            print(f"[synthesize] → {base_name} ({full_bytes / 1024:.1f}KB)")
        else:
            # 分卷：按知识单元边界切分
            volume_files: List[str] = []
            current_parts: List[str] = []
            current_units: List[KnowledgeUnit] = []
            current_size = 0
            vol_num = 1

            for u in units:
                u_bytes = len(u.body.encode("utf-8"))
                if current_parts and current_size + u_bytes + 2 > max_bytes:
                    # 写入当前卷
                    vol_name = base_name if vol_num == 1 else f"{safe_name}-{vol_num:03d}.md"
                    vol_path = topics_dir / vol_name
                    _atomic_write(vol_path, "\n\n".join(current_parts))
                    volume_files.append(vol_name)
                    for cu in current_units:
                        unit_file_map[cu.fingerprint] = vol_name
                    print(f"[synthesize] → {vol_name} ({current_size / 1024:.1f}KB)")
                    current_parts = []
                    current_units = []
                    current_size = 0
                    vol_num += 1

                current_parts.append(u.body)
                current_units.append(u)
                current_size += u_bytes + 2

            if current_parts:
                vol_name = base_name if vol_num == 1 else f"{safe_name}-{vol_num:03d}.md"
                vol_path = topics_dir / vol_name
                _atomic_write(vol_path, "\n\n".join(current_parts))
                volume_files.append(vol_name)
                for cu in current_units:
                    unit_file_map[cu.fingerprint] = vol_name
                print(f"[synthesize] → {vol_name} ({current_size / 1024:.1f}KB)")

            written_files.extend(volume_files)
            topic_fp_cache[topic_name] = {"fingerprint": content_fp, "files": volume_files}

    return written_files, unit_file_map


def generate_index(
    topics_dir: Path,
    topic_units: Dict[str, List[KnowledgeUnit]],
    unmapped_tags: List[str],
    tag_buckets: Dict[str, List[KnowledgeUnit]],
    mapping: Dict[str, List[str]],
) -> None:
    """生成 00-目录与导读.md。"""
    lines = [
        "# 网文新人指南\n",
        "> 从零开始写网文的系统学习路径\n",
        "## 目录\n",
        "| 序号 | 主题 | 知识单元 | P0 | P1 | P2 | 包含标签 |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]

    sorted_topics = sorted(
        [(k, v) for k, v in topic_units.items() if k != "未分类"],
        key=lambda kv: kv[0],
    )

    for topic_name, units in sorted_topics:
        seq = topic_name.split("-")[0] if "-" in topic_name else "?"
        total = len(units)
        p0 = sum(1 for u in units if u.priority == 0)
        p1 = sum(1 for u in units if u.priority == 1)
        p2 = sum(1 for u in units if u.priority >= 2)
        tags = mapping.get(topic_name, [])
        tags_str = "、".join(tags[:6]) + ("..." if len(tags) > 6 else "")
        lines.append(f"| {seq} | {topic_name} | {total} 条 | {p0} | {p1} | {p2} | {tags_str} |")

    lines.append("")
    lines.append("## 如何使用本指南\n")
    lines.append("- 01-03：写书前必读（认知、选题、开篇）")
    lines.append("- 04-06：写作过程中反复查阅（元素、节奏、规划）")
    lines.append("- 07-08：遇到困难时翻阅（避坑、心态）")

    if unmapped_tags:
        lines.append("")
        lines.append("## 未分类标签\n")
        lines.append("以下标签未被映射到任何主题，请检查 config.yaml 的 mapping：")
        for tag in sorted(unmapped_tags):
            count = len(tag_buckets.get(tag, []))
            lines.append(f"- {tag}（{count} 条知识单元）")

    index_path = topics_dir / "00-目录与导读.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[synthesize] → 00-目录与导读.md")


def generate_cross_topic_report(
    cross_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]],
    unit_topic: Dict[str, str],
    intermediate_dir: Path,
) -> None:
    """生成跨主题相似度报告。"""
    if not cross_pairs:
        print("[synthesize] 无跨主题相似对，跳过报告")
        return

    lines = ["跨主题相似度报告（阈值: 70%）\n"]

    sorted_pairs = sorted(cross_pairs, key=lambda x: x[2], reverse=True)
    for a, b, sim in sorted_pairs[:50]:  # 最多报告 50 对
        topic_a = unit_topic.get(a.fingerprint, "?")
        topic_b = unit_topic.get(b.fingerprint, "?")
        preview_a = a.body[:80].replace("\n", " ")
        preview_b = b.body[:80].replace("\n", " ")
        lines.append(f"[{sim:.0%}] ⚠ 跨主题:")
        lines.append(f"  主题A: {topic_a} | 标签: {a.tag} | P{a.priority} | \"{preview_a}\"")
        lines.append(f"  主题B: {topic_b} | 标签: {b.tag} | P{b.priority} | \"{preview_b}\"")
        lines.append(f"  → 建议检查 mapping 是否需要调整")
        lines.append("")

    report_path = intermediate_dir / "cross_topic_similarity.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[synthesize] → 跨主题相似度报告: {len(sorted_pairs[:50])} 对")


def generate_changelog(
    new_fps: set,
    all_units: List[KnowledgeUnit],
    unit_file_map: Dict[str, str],
    unit_topic_map: Dict[str, str],
    intermediate_dir: Path,
) -> None:
    """生成本次 synthesize 新增知识单元的变更报告。"""
    if not new_fps:
        print("[synthesize] 无新增知识单元，跳过变更报告")
        return

    # 筛选新增单元，且仅保留本次写入了文件的（排除缓存跳过的主题）
    new_units = [u for u in all_units if u.fingerprint in new_fps and u.fingerprint in unit_file_map]
    if not new_units:
        print("[synthesize] 新增单元均属于缓存跳过的主题，跳过变更报告")
        return

    # 按主题分组
    by_topic: Dict[str, List[KnowledgeUnit]] = {}
    for u in new_units:
        topic = unit_topic_map.get(u.fingerprint, "未分类")
        by_topic.setdefault(topic, []).append(u)

    now = dt.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ts_display = now.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Synthesize 变更报告\n",
        f"> 生成时间: {ts_display}",
        f"> 新增知识单元: {len(new_units)} 条\n",
    ]

    for topic_name in sorted(by_topic.keys()):
        topic_units = by_topic[topic_name]
        lines.append(f"## {topic_name} (+{len(topic_units)})\n")
        for u in topic_units:
            fname = unit_file_map[u.fingerprint]
            lines.append(f"- [{fname}](../topics/{fname}) — [{u.tag}][P{u.priority}] {u.title}")
        lines.append("")

    changelog_path = intermediate_dir / f"changelog_{timestamp}.md"
    changelog_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[synthesize] → 变更报告: {len(new_units)} 条新增知识单元 ({changelog_path.name})")


def run_synthesize_stage(
    config: PipelineConfig,
    extract_dir: Path,
    full_resync: bool = False,
) -> None:
    """Synthesize 阶段：六阶段流水线。

    Phase 1: 结构化解析 + 归组（纯代码）
    Phase 2: 增量相似度扫描（embedding 余弦相似度 / n-gram Jaccard 降级）
    Phase 3: 疑似重复对送 LLM 判断（仅限同主题内 ≥ threshold）
    Phase 4: 内存中去重/合并
    Phase 5: 拼接写入文件（含 100KB 分卷）
    Phase 6: 生成 00-目录与导读.md + 跨主题相似度报告
    """

    log_dir = config.output_dir / "_logs"
    endpoint_states = _create_endpoint_states(config, config.synthesize_stage, log_dir=log_dir)
    intermediate_dir = config.output_dir / "_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    topics_dir = config.output_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = config.merge.similarity
    mapping = config.merge.mapping

    # ── 缓存检查（基于 extract 指纹 + merge 配置签名） ──
    fingerprint_path = intermediate_dir / "synthesize_fingerprint.txt"
    manifest_path = intermediate_dir / "synthesize_manifest.json"
    extract_fp = _compute_extract_fingerprint(extract_dir)
    merge_sig = hashlib.sha256(
        json.dumps({
            "mapping": mapping,
            "similarity": {
                "threshold": sim_cfg.threshold,
                "ngram_size": sim_cfg.ngram_size,
                "method": sim_cfg.method,
                "embedding_model": sim_cfg.embedding_model,
            },
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    current_fingerprint = f"{extract_fp}_{merge_sig}"

    if not full_resync and fingerprint_path.is_file() and manifest_path.is_file():
        stored_fp = fingerprint_path.read_text(encoding="utf-8").strip()
        if stored_fp == current_fingerprint:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                topic_files = manifest.get("topic_files", [])
                if topic_files and all((topics_dir / f).is_file() for f in topic_files):
                    print("[synthesize] extract 指纹匹配且所有主题文件存在，跳过")
                    print(f"[synthesize] 阶段完成，主题文档目录：{topics_dir}")
                    return
            except (json.JSONDecodeError, KeyError):
                pass

    # ── 加载缓存 ──
    unit_fp_path = intermediate_dir / "unit_fingerprints.json"
    dedup_cache_path = intermediate_dir / "dedup_judgments.json"
    topic_fp_path = intermediate_dir / "synthesize_topic_fingerprints.json"

    cached_unit_fps: set = set()
    if not full_resync and unit_fp_path.is_file():
        try:
            fp_data = json.loads(unit_fp_path.read_text(encoding="utf-8"))
            if fp_data.get("merge_sig") == merge_sig:
                cached_unit_fps = set(fp_data.get("units", {}).keys())
        except (json.JSONDecodeError, KeyError):
            pass

    cached_judgments: Dict[str, dict] = {}
    cached_dedup_sig = ""
    if not full_resync and dedup_cache_path.is_file():
        try:
            dedup_data = json.loads(dedup_cache_path.read_text(encoding="utf-8"))
            cached_dedup_sig = dedup_data.get("merge_sig", "")
            if cached_dedup_sig == merge_sig:
                cached_judgments = dedup_data.get("judgments", {})
        except (json.JSONDecodeError, KeyError):
            pass

    topic_fp_cache: dict = {}
    if not full_resync and topic_fp_path.is_file():
        try:
            fp_data = json.loads(topic_fp_path.read_text(encoding="utf-8"))
            if fp_data.get("merge_sig") == merge_sig:
                topic_fp_cache = fp_data.get("topics", {})
        except (json.JSONDecodeError, KeyError):
            pass

    # ══════════════════════════════════════════════════════════
    # Phase 1: 结构化解析 + 归组（纯代码）
    # ══════════════════════════════════════════════════════════
    print("[synthesize] Phase 1: 结构化解析 + 归组...")
    tag_buckets, all_units = group_by_category(extract_dir)
    total_units = len(all_units)
    print(f"[synthesize] 共 {total_units} 条知识单元，归入 {len(tag_buckets)} 个标签")

    # 标签→主题映射
    if mapping:
        print("[synthesize] 使用 config.yaml 手动标签映射")
        topic_units, unmapped_tags = apply_topic_mapping(tag_buckets, mapping)
    else:
        print("[synthesize] 无映射配置，每个标签独立成主题")
        topic_units = {k: v for k, v in tag_buckets.items()}
        unmapped_tags = []

    # 排序显示
    topic_names = sorted(topic_units.keys(), key=lambda k: (k == "未分类", k))
    print(f"[synthesize] 合并后 {len(topic_names)} 个主题：")
    for name in topic_names:
        units = topic_units[name]
        total_kb = sum(len(u.body.encode("utf-8")) for u in units) / 1024.0
        p_counts = {0: 0, 1: 0, 2: 0}
        for u in units:
            if u.priority in p_counts:
                p_counts[u.priority] += 1
        print(f"  - {name}: {len(units)} 条 (P0={p_counts[0]}, P1={p_counts[1]}, P2={p_counts[2]}), {total_kb:.1f}KB")

    if unmapped_tags:
        print(f"[synthesize] ⚠ 未映射标签 ({len(unmapped_tags)}): {', '.join(unmapped_tags[:10])}")

    # 保存 unit 指纹缓存
    unit_fp_data = {
        "merge_sig": merge_sig,
        "units": {},
    }
    unit_topic_map: Dict[str, str] = {}
    for topic_name, units in topic_units.items():
        for u in units:
            unit_fp_data["units"][u.fingerprint] = {
                "tag": u.tag, "topic": topic_name, "priority": u.priority,
            }
            unit_topic_map[u.fingerprint] = topic_name
    dump_json(unit_fp_path, unit_fp_data)

    # ══════════════════════════════════════════════════════════
    # Phase 2: 增量相似度扫描
    # ══════════════════════════════════════════════════════════
    all_fps = {u.fingerprint for u in all_units}
    new_fps = all_fps - cached_unit_fps
    print(f"[synthesize] 全量 {len(all_fps)} 条, 新增 {len(new_fps)} 条, 缓存 {len(cached_unit_fps)} 条")

    if not new_fps and cached_unit_fps:
        print("[synthesize] 无新增单元，跳过相似度扫描")
        intra_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []
        cross_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []
    elif sim_cfg.method == "embedding":
        emb_src = sim_cfg.embedding_api_url if sim_cfg.embedding_api_url else "synthesize 端点"
        print(f"[synthesize] Phase 2: Embedding 语义相似度扫描 (阈值={sim_cfg.threshold:.0%}, model={sim_cfg.embedding_model}, 端点={emb_src})...")

        # 1. 加载 embedding 缓存
        emb_cache_path = intermediate_dir / "embedding_cache.json"
        emb_cache = _load_embedding_cache(emb_cache_path) if not full_resync else {}
        cached_emb_count = len(emb_cache)

        # 2. 找出未缓存的单元，批量调用 embedding API
        uncached_units = [u for u in all_units if u.fingerprint not in emb_cache]
        if uncached_units:
            print(f"[embedding] 需要获取 {len(uncached_units)} 条 embedding (已缓存 {cached_emb_count} 条)...")
            texts = [u.body for u in uncached_units]
            vectors = _get_embeddings(
                texts, config, endpoint_states, sim_cfg,
            )
            for u, vec in zip(uncached_units, vectors):
                emb_cache[u.fingerprint] = vec
            print(f"[embedding] 获取完成, 总缓存 {len(emb_cache)} 条")

            # 3. 保存缓存
            _save_embedding_cache(emb_cache_path, emb_cache)
        else:
            print(f"[embedding] 全部 {cached_emb_count} 条 embedding 已缓存")

        # 4. 调用 embedding 相似度扫描
        intra_pairs, cross_pairs = scan_similarity_embedding(
            topic_units, all_units, cached_unit_fps,
            sim_cfg.threshold, emb_cache,
        )
    else:
        print(f"[synthesize] Phase 2: 增量相似度扫描 n-gram (阈值={sim_cfg.threshold:.0%}, n={sim_cfg.ngram_size})...")
        intra_pairs, cross_pairs = scan_similarity(
            topic_units, all_units, cached_unit_fps,
            sim_cfg.threshold, sim_cfg.ngram_size,
        )
    print(f"[synthesize] 同主题相似对: {len(intra_pairs)}, 跨主题相似对: {len(cross_pairs)}")

    # ══════════════════════════════════════════════════════════
    # Phase 3a: 批量 verdict 判断（仅同主题内）
    # ══════════════════════════════════════════════════════════
    print(f"[synthesize] Phase 3a: LLM verdict 判断...")

    # 过滤已有缓存的判断
    new_pairs: List[tuple[KnowledgeUnit, KnowledgeUnit, float]] = []
    for a, b, sim in intra_pairs:
        pair_key = "_".join(sorted([a.fingerprint, b.fingerprint]))
        if pair_key not in cached_judgments:
            new_pairs.append((a, b, sim))

    print(f"[synthesize] 需要 LLM 判断: {len(new_pairs)} 对 (已缓存: {len(intra_pairs) - len(new_pairs)} 对)")

    # 批量送 LLM 获取 verdict
    if new_pairs:
        batch_size = sim_cfg.llm_batch_size
        system_prompt = config.synthesize_stage.system_prompt
        consecutive_all_fail = 0

        for batch_start in range(0, len(new_pairs), batch_size):
            batch = new_pairs[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(new_pairs) + batch_size - 1) // batch_size
            print(f"[synthesize] verdict 批次 {batch_num}/{total_batches}: {len(batch)} 对")

            prompt = build_dedup_prompt(batch)
            try:
                raw = call_llm_with_failover(
                    stage=config.synthesize_stage,
                    endpoint_states=endpoint_states,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    runtime=config.runtime,
                )
                verdicts = _parse_verdicts(raw, len(batch))
                consecutive_all_fail = 0

                for pair_idx, entry in enumerate(verdicts):
                    a, b, sim = batch[pair_idx]
                    pair_key = "_".join(sorted([a.fingerprint, b.fingerprint]))
                    verdict = entry.get("verdict", "KEEP_BOTH")
                    judgment: dict = {"verdict": verdict}

                    if verdict == "DUPLICATE":
                        keep_side = entry.get("keep", "A")
                        if keep_side == "B":
                            judgment["keep"] = b.fingerprint
                            judgment["drop"] = a.fingerprint
                        else:
                            judgment["keep"] = a.fingerprint
                            judgment["drop"] = b.fingerprint
                    # MERGE: 此时仅记录 verdict，merged_text 留到 Phase 3b

                    cached_judgments[pair_key] = judgment

            except EndpointsExhaustedError:
                print("[synthesize] 所有端点熔断，停止 verdict 判断")
                break
            except PipelineError as exc:
                consecutive_all_fail += 1
                print(f"[synthesize] verdict 判断失败: {exc}")
                if consecutive_all_fail >= 3:
                    print("[synthesize] 连续 3 次失败，停止 verdict 判断")
                    break

    # 中间保存一次缓存（verdict 结果）
    dump_json(dedup_cache_path, {
        "merge_sig": merge_sig,
        "judgments": cached_judgments,
    })

    dup_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "DUPLICATE")
    merge_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "MERGE")
    keep_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "KEEP_BOTH")
    print(f"[synthesize] verdict 结果: DUPLICATE={dup_count}, MERGE={merge_count}, KEEP_BOTH={keep_count}")

    # ══════════════════════════════════════════════════════════
    # Phase 3b: 逐对合并（仅处理 MERGE 且缺少 merged_text 的对）
    # ══════════════════════════════════════════════════════════
    merge_needed: List[Tuple[str, KnowledgeUnit, KnowledgeUnit, float]] = []
    for a, b, sim in intra_pairs:
        pair_key = "_".join(sorted([a.fingerprint, b.fingerprint]))
        j = cached_judgments.get(pair_key)
        if j and j.get("verdict") == "MERGE" and not j.get("merged_text"):
            merge_needed.append((pair_key, a, b, sim))

    if merge_needed:
        print(f"[synthesize] Phase 3b: 逐对合并 {len(merge_needed)} 对...")
        system_prompt = config.synthesize_stage.system_prompt
        consecutive_merge_fail = 0

        for mi, (pair_key, a, b, sim) in enumerate(merge_needed, 1):
            print(f"[synthesize] 合并 {mi}/{len(merge_needed)}: [{a.tag}] {a.title} ↔ {b.title}")
            prompt = build_merge_prompt(a, b, sim)
            try:
                merged_text = call_llm_with_continuation(
                    stage=config.synthesize_stage,
                    endpoint_states=endpoint_states,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    runtime=config.runtime,
                )
                merged_text = strip_code_fences(merged_text).strip()
                if merged_text:
                    cached_judgments[pair_key]["merged_text"] = merged_text
                    consecutive_merge_fail = 0
                else:
                    print(f"[synthesize] 合并返回空文本，跳过")
            except EndpointsExhaustedError:
                print("[synthesize] 所有端点熔断，停止合并")
                break
            except PipelineError as exc:
                consecutive_merge_fail += 1
                print(f"[synthesize] 合并失败: {exc}")
                if consecutive_merge_fail >= 3:
                    print("[synthesize] 连续 3 次合并失败，停止")
                    break

    # 保存最终判断缓存
    dump_json(dedup_cache_path, {
        "merge_sig": merge_sig,
        "judgments": cached_judgments,
    })

    dup_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "DUPLICATE")
    merge_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "MERGE" and j.get("merged_text"))
    merge_pending = sum(1 for j in cached_judgments.values() if j.get("verdict") == "MERGE" and not j.get("merged_text"))
    keep_count = sum(1 for j in cached_judgments.values() if j.get("verdict") == "KEEP_BOTH")
    print(f"[synthesize] 最终判断: DUPLICATE={dup_count}, MERGE={merge_count} (未合并={merge_pending}), KEEP_BOTH={keep_count}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: 内存中去重/合并
    # ══════════════════════════════════════════════════════════
    print("[synthesize] Phase 4: 去重/合并...")
    deduped_units = apply_dedup_verdicts(topic_units, cached_judgments)

    total_before = sum(len(v) for v in topic_units.values())
    total_after = sum(len(v) for v in deduped_units.values())
    print(f"[synthesize] 去重前: {total_before} 条, 去重后: {total_after} 条 (减少 {total_before - total_after} 条)")

    # ══════════════════════════════════════════════════════════
    # Phase 5: 拼接写入文件（含 100KB 分卷）
    # ══════════════════════════════════════════════════════════
    print(f"[synthesize] Phase 5: 写入主题文件 (分卷阈值: {sim_cfg.max_file_size_kb}KB)...")

    # 清理 topics 目录中不属于当前 mapping 的旧文件
    valid_safe_names = {safe_filename(tn) for tn in deduped_units}
    for old_file in topics_dir.glob("*.md"):
        if old_file.name == "00-目录与导读.md":
            continue
        # 检查文件是否属于当前有效主题（精确匹配 safe_name.md 或 safe_name-NNN.md）
        stem = old_file.stem  # e.g. "01-入门认知篇" or "01-入门认知篇-002"
        belongs = False
        for sn in valid_safe_names:
            if stem == sn or (stem.startswith(sn + "-") and stem[len(sn) + 1:].isdigit()):
                belongs = True
                break
        if not belongs:
            old_file.unlink()
            print(f"[synthesize] 清理旧文件: {old_file.name}")

    topic_files, unit_file_map = write_topic_files(
        deduped_units, topics_dir, sim_cfg.max_file_size_kb, topic_fp_cache,
    )

    # 保存 topic 指纹缓存
    _save_topic_fingerprints(topic_fp_path, merge_sig, topic_fp_cache)

    # ══════════════════════════════════════════════════════════
    # Phase 6: 生成 00-目录与导读.md + 跨主题相似度报告
    # ══════════════════════════════════════════════════════════
    print("[synthesize] Phase 6: 生成目录与报告...")
    generate_index(topics_dir, deduped_units, unmapped_tags, tag_buckets, mapping)
    generate_cross_topic_report(cross_pairs, unit_topic_map, intermediate_dir)

    # ── 变更报告（首次运行不生成） ──
    if cached_unit_fps:
        generate_changelog(
            new_fps=new_fps,
            all_units=all_units,
            unit_file_map=unit_file_map,
            unit_topic_map=unit_topic_map,
            intermediate_dir=intermediate_dir,
        )

    # ── 写入缓存 manifest ──
    topic_files.sort()
    manifest_payload = {
        "fingerprint": current_fingerprint,
        "topic_count": len(topic_files),
        "topics_dir": str(topics_dir),
        "topic_files": topic_files,
        "mapping": {k: v for k, v in mapping.items()},
    }
    dump_json(manifest_path, manifest_payload)
    fingerprint_path.write_text(current_fingerprint, encoding="utf-8")

    # ── 写入 run_summary ──
    summary_payload = {
        "topic_count": len(topic_files),
        "topics_dir": str(topics_dir),
        "topic_files": topic_files,
        "dedup_stats": {
            "total_before": total_before,
            "total_after": total_after,
            "duplicate": dup_count,
            "merge": merge_count,
            "keep_both": keep_count,
            "cross_topic_pairs": len(cross_pairs),
        },
    }
    dump_json(intermediate_dir / "run_summary.json", summary_payload)

    print(f"[synthesize] 阶段完成，共 {len(topic_files)} 个主题文件，目录：{topics_dir}")



# ── Pipeline 入口 ──


def _fmt_endpoints(stage: StageConfig) -> str:
    """格式化端点信息用于日志输出。"""
    if stage.endpoints:
        parts = [
            f"{ep.name}({ep.api_url})"
            + (f"[{ep.api_type}]" if ep.api_type else "")
            for ep in stage.endpoints
        ]
        return " → ".join(parts)
    return stage.api_url or "default"


def run_pipeline(config: PipelineConfig, only_stage: str, full_resync: bool = False) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] 配置加载完成")
    print(f"[pipeline] 输入目录：{config.input_dir}")
    print(f"[pipeline] 输出目录：{config.output_dir}")
    print(
        f"[pipeline] extract: {config.extract_stage.provider}/{config.extract_stage.model} | "
        f"synthesize: {config.synthesize_stage.provider}/{config.synthesize_stage.model}"
    )
    print(f"[pipeline] extract 端点: {_fmt_endpoints(config.extract_stage)}")
    print(f"[pipeline] synthesize 端点: {_fmt_endpoints(config.synthesize_stage)}")
    workers = config.concurrency.max_workers
    print(f"[pipeline] 并发: max_workers={workers}" + (" (串行模式)" if workers <= 1 else ""))

    extract_dir: Path | None = None

    if only_stage in {"all", "extract"}:
        extract_dir = run_extract_stage(config)

    if only_stage in {"all", "synthesize"}:
        if extract_dir is None:
            extract_dir = get_intermediate_extract_dir(config.output_dir)
        run_synthesize_stage(config, extract_dir, full_resync)


def resolve_error_log_dir(config_path: Path, config: PipelineConfig | None) -> Path:
    if config is not None:
        return config.output_dir / "_logs"
    return config_path.parent / "_logs"


def write_error_log(
    log_dir: Path,
    config_path: Path,
    only_stage: str,
    exc: BaseException,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_error_{timestamp}.log"

    lines = [
        f"时间: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.executable}",
        f"工作目录: {Path.cwd()}",
        f"配置文件: {config_path}",
        f"执行阶段: {only_stage}",
        f"异常类型: {type(exc).__name__}",
        f"异常信息: {exc}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config: PipelineConfig | None = None

    try:
        config = load_config(config_path)
        run_pipeline(config, args.only_stage, args.full_resync)
        print("[pipeline] 全部执行完成")
    except KeyboardInterrupt as exc:
        log_dir = resolve_error_log_dir(config_path, config)
        log_path = write_error_log(log_dir, config_path, args.only_stage, exc)
        print("\n[ERROR] 用户中断执行")
        print(f"[ERROR] 错误日志已写入：{log_path}")
        sys.exit(130)
    except Exception as exc:
        log_dir = resolve_error_log_dir(config_path, config)
        log_path = write_error_log(log_dir, config_path, args.only_stage, exc)
        print(f"[ERROR] {exc}")
        print(f"[ERROR] 错误日志已写入：{log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
