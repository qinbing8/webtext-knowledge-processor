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


@dataclass(frozen=True)
class SynthesizeMergeConfig:
    target_categories: int = 15
    use_llm: bool = True
    mapping: Dict[str, List[str]] = field(default_factory=dict)


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

    merge_config = SynthesizeMergeConfig(
        target_categories=int(grouping_raw.get("target_categories", 15)),
        use_llm=bool(grouping_raw.get("use_llm", True)),
        mapping=grouping_mapping,
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
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if extra_messages:
        for em in extra_messages:
            messages.append({
                "role": em["role"],
                "content": [{"type": "text", "text": em["content"]}],
            })
    payload = {
        "model": model,
        "system": [{"type": "text", "text": system_prompt}],
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
            next_i = i + 1
            if next_i < len(available):
                print(f"[failover] 端点 '{es.name}' 失败，切换到 '{available[next_i].name}'")

    raise PipelineError("所有端点均调用失败：" + " | ".join(failures))


# ── 自动续接 ──


_CONTINUATION_MAX_ROUNDS = 5
_CONTINUATION_END_MARKER = "<!-- END -->"


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


def _save_topic_fingerprints(path: Path, merge_sig: str, fps: dict) -> None:
    """持久化 per-topic 指纹缓存。"""
    dump_json(path, {"merge_sig": merge_sig, "topics": fps})


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

    if max_workers <= 1:
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
    if max_workers > 1:
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


def group_by_category(extract_dir: Path) -> Dict[str, List[str]]:
    """读取 extract 目录所有 .md，按 ## [分类标签][Px] 标签归组知识单元。

    返回 { 分类名: [知识单元完整文本, ...] }。
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

    tag_re = re.compile(r'^##\s*\[([^\]]+)\]\s*\[P[012]\]\s*')

    buckets: Dict[str, List[str]] = {}

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        # 按 ## (非 ###) 边界拆分知识单元
        units = re.split(r'\n(?=## (?!#))', text)

        for unit in units:
            unit = unit.strip()
            if not unit or not unit.startswith("## "):
                # 跳过文件级标题(#)、前言、分隔线等非知识单元内容
                continue

            first_line = unit.split("\n", 1)[0]
            match = tag_re.match(first_line)
            category = match.group(1).strip() if match else "未分类"
            buckets.setdefault(category, []).append(unit)

    return buckets


def run_synthesize_stage(
    config: PipelineConfig,
    extract_dir: Path,
) -> None:
    """Synthesize 阶段：按分类标签归组 → 标签合并 → 逐分类 LLM 合并去重。"""

    log_dir = config.output_dir / "_logs"
    endpoint_states = _create_endpoint_states(config, config.synthesize_stage, log_dir=log_dir)
    intermediate_dir = config.output_dir / "_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    topics_dir = config.output_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    # ── 缓存检查（基于 extract 指纹 + merge 配置签名） ──
    fingerprint_path = intermediate_dir / "synthesize_fingerprint.txt"
    manifest_path = intermediate_dir / "synthesize_manifest.json"
    extract_fp = _compute_extract_fingerprint(extract_dir)
    merge_sig = hashlib.sha256(
        json.dumps({
            "target_categories": config.merge.target_categories,
            "use_llm": config.merge.use_llm,
            "mapping": config.merge.mapping,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    current_fingerprint = f"{extract_fp}_{merge_sig}"

    if fingerprint_path.is_file() and manifest_path.is_file():
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

    # ── per-topic 指纹缓存（增量模式：仅重做输入变化的主题） ──
    topic_fp_path = intermediate_dir / "synthesize_topic_fingerprints.json"
    cached_topic_fps: dict[str, str] = {}
    if topic_fp_path.is_file():
        try:
            fp_data = json.loads(topic_fp_path.read_text(encoding="utf-8"))
            # merge 配置变更 → 全部失效
            if fp_data.get("merge_sig") == merge_sig:
                cached_topic_fps = fp_data.get("topics", {})
        except (json.JSONDecodeError, KeyError):
            pass

    # 提前写入指纹，使全局 manifest 与当前指纹关联
    fingerprint_path.write_text(current_fingerprint, encoding="utf-8")

    # ── Phase 1: 按分类标签归组（纯代码，零信息损失） ──
    print("[synthesize] Phase 1: 按分类标签归组知识单元...")
    buckets = group_by_category(extract_dir)

    total_units = sum(len(v) for v in buckets.values())
    print(f"[synthesize] 共 {total_units} 条知识单元，归入 {len(buckets)} 个标签")

    # ── Phase 1.5: 标签合并 ──
    print("[synthesize] Phase 1.5: 标签合并...")
    mapping = _resolve_tag_mapping(
        list(buckets.keys()), config.merge,
        config.synthesize_stage, endpoint_states, config.runtime,
    )
    merged_buckets = _apply_tag_grouping(buckets, mapping)
    mapping_mode = "manual" if config.merge.mapping else ("llm" if config.merge.use_llm else "identity")

    # 排序：未分类放最后
    topic_names = sorted(merged_buckets.keys(), key=lambda k: (k == "未分类", k))

    print(f"[synthesize] 合并后 {len(topic_names)} 个分类：")
    for name in topic_names:
        units = merged_buckets[name]
        total_kb = sum(len(u.encode("utf-8")) for u in units) / 1024.0
        print(f"  - {name}: {len(units)} 条, {total_kb:.1f}KB")

    # ── Phase 2: 逐分类 LLM 合并去重 ──
    synth_cfg = config.batching.synthesize
    effective_limit = get_effective_size_limit(
        synth_cfg.max_batch_size_kb,
        config.synthesize_stage.provider,
        config.batching.provider_limits,
    )
    synthesize_system_prompt = config.synthesize_stage.system_prompt

    topic_files: List[str] = []
    total = len(topic_names)

    # ── 并发执行基础设施 ──
    manifest_lock = threading.Lock()
    stop_event = threading.Event()
    topic_files_lock = threading.Lock()
    synth_worker_id_counter = [0]
    synth_worker_id_lock = threading.Lock()
    synth_clog = _ConcurrentLog("synthesize", config.concurrency.max_workers, total)

    def _process_topic(i: int, topic_name: str) -> str | None:
        """处理单个主题（线程安全），返回输出文件名或 None。"""
        if stop_event.is_set():
            synth_clog.record(
                worker=threading.current_thread().name,
                unit=topic_name,
                status="skip",
                duration_s=0.0,
            )
            return None

        units = merged_buckets[topic_name]
        combined_text = "\n\n---\n\n".join(units)
        combined_kb = len(combined_text.encode("utf-8")) / 1024.0

        filename = f"{i:02d}_{safe_filename(topic_name)}.md"
        topic_path = topics_dir / filename

        # 为并发模式分配 worker ID
        s_max_workers = config.concurrency.max_workers
        s_tag = ""
        if s_max_workers > 1:
            with synth_worker_id_lock:
                swid = synth_worker_id_counter[0]
                synth_worker_id_counter[0] += 1
            s_tag = f" W{swid % s_max_workers}"

        t0 = time.time()
        worker_name = threading.current_thread().name

        # 增量缓存：主题输入未变则跳过
        current_fp = _compute_topic_fingerprint(units)
        if (topic_name in cached_topic_fps and
                cached_topic_fps[topic_name] == current_fp and
                topic_path.is_file()):
            print(f"[synthesize{s_tag} {i}/{total}] {topic_name} (内容未变，跳过)")
            synth_clog.record(worker=worker_name, unit=topic_name, status="skip", duration_s=0.0)
            return filename

        print(f"[synthesize{s_tag} {i}/{total}] {topic_name} ({len(units)} 条, {combined_kb:.1f}KB)")

        if len(units) == 1:
            print(f"[synthesize{s_tag}] → 仅 1 条，直接写入: {filename}")
            topic_path.write_text(units[0], encoding="utf-8")
            with manifest_lock:
                cached_topic_fps[topic_name] = current_fp
                _save_topic_fingerprints(topic_fp_path, merge_sig, cached_topic_fps)
            synth_clog.record(worker=worker_name, unit=topic_name, status="ok", duration_s=time.time() - t0)
            return filename

        user_prompt = (
            f"以下是关于【{topic_name}】这一主题的 {len(units)} 条知识单元，"
            f"来自不同来源的提取结果。请按规则合并整理，保留所有独特内容。\n\n"
            + combined_text
        )

        try:
            if combined_kb <= effective_limit:
                raw = call_llm_with_continuation(
                    stage=config.synthesize_stage,
                    endpoint_states=endpoint_states,
                    system_prompt=synthesize_system_prompt,
                    user_prompt=user_prompt,
                    runtime=config.runtime,
                )
                result = strip_code_fences(raw)
                topic_path.write_text(result, encoding="utf-8")
                result_kb = len(result.encode("utf-8")) / 1024.0
                print(f"[synthesize{s_tag}] → {filename} ({result_kb:.1f}KB)")
            else:
                print(
                    f"[synthesize{s_tag}] 主题过大 ({combined_kb:.1f}KB > {effective_limit}KB)，分批处理"
                )
                sub_batches = compute_batches(
                    [{"text": u} for u in units],
                    999,
                    effective_limit,
                )

                sub_results: List[str] = []
                for si, sub_batch in enumerate(sub_batches, 1):
                    if stop_event.is_set():
                        synth_clog.record(worker=worker_name, unit=topic_name, status="skip", duration_s=time.time() - t0)
                        return None
                    sub_text = "\n\n---\n\n".join(item["text"] for item in sub_batch)
                    sub_kb = len(sub_text.encode("utf-8")) / 1024.0
                    print(f"[synthesize{s_tag}]   子批 {si}/{len(sub_batches)}: {sub_kb:.1f}KB")

                    sub_prompt = (
                        f"以下是关于【{topic_name}】这一主题的部分知识单元"
                        f"（第 {si}/{len(sub_batches)} 批）。"
                        f"请合并整理，保留所有独特内容。\n\n"
                        + sub_text
                    )

                    try:
                        raw = call_llm_with_continuation(
                            stage=config.synthesize_stage,
                            endpoint_states=endpoint_states,
                            system_prompt=synthesize_system_prompt,
                            user_prompt=sub_prompt,
                            runtime=config.runtime,
                        )
                        sub_results.append(strip_code_fences(raw))
                    except PipelineError as exc:
                        if isinstance(exc, EndpointsExhaustedError):
                            raise
                        print(f"[synthesize{s_tag}]   子批 {si} 失败: {exc}，保留原文")
                        sub_results.append(sub_text)

                final_text = "\n\n---\n\n".join(sub_results)
                topic_path.write_text(final_text, encoding="utf-8")
                final_kb = len(final_text.encode("utf-8")) / 1024.0
                print(f"[synthesize{s_tag}] → {filename} ({final_kb:.1f}KB)")

            synth_clog.record(worker=worker_name, unit=topic_name, status="ok", duration_s=time.time() - t0)

        except EndpointsExhaustedError:
            synth_clog.record(worker=worker_name, unit=topic_name, status="error", duration_s=time.time() - t0, error="EndpointsExhausted")
            stop_event.set()
            raise
        except PipelineError as exc:
            synth_clog.record(worker=worker_name, unit=topic_name, status="error", duration_s=time.time() - t0, error=str(exc))
            print(f"[synthesize{s_tag}] 失败: {topic_name}: {exc}")
            topic_path.write_text(combined_text, encoding="utf-8")
            print(f"[synthesize{s_tag}] → 降级为原文拼接: {filename}")

        with manifest_lock:
            cached_topic_fps[topic_name] = current_fp
            _save_topic_fingerprints(topic_fp_path, merge_sig, cached_topic_fps)
        return filename

    # ── 执行（串行 or 并发） ──
    max_workers = config.concurrency.max_workers

    if max_workers <= 1:
        for i, topic_name in enumerate(topic_names, 1):
            result = _process_topic(i, topic_name)
            if result:
                topic_files.append(result)
            if stop_event.is_set():
                break
    else:
        print(f"[synthesize] 并发模式：max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="synth") as executor:
            futures = {}
            for i, topic_name in enumerate(topic_names, 1):
                f = executor.submit(_process_topic, i, topic_name)
                futures[f] = topic_name
            try:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        for pending_f in futures:
                            pending_f.cancel()
                        break
                    try:
                        result = future.result()
                        if result:
                            with topic_files_lock:
                                topic_files.append(result)
                    except EndpointsExhaustedError:
                        for pending_f in futures:
                            pending_f.cancel()
                        break
            except KeyboardInterrupt:
                print("\n[synthesize] 用户中断，取消剩余任务...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    # ── 并发日志输出 ──
    if max_workers > 1:
        synth_clog.write_log(log_dir)
        synth_clog.print_summary()

    # 排序确保文件列表顺序一致
    topic_files.sort()

    # ── 写入缓存 manifest ──
    manifest_payload = {
        "fingerprint": current_fingerprint,
        "topic_count": len(topic_files),
        "topics_dir": str(topics_dir),
        "topic_files": topic_files,
        "mapping": {k: v for k, v in mapping.items()},
        "mapping_mode": mapping_mode,
    }
    dump_json(manifest_path, manifest_payload)
    fingerprint_path.write_text(current_fingerprint, encoding="utf-8")

    # ── 写入 run_summary ──
    summary_payload = {
        "topic_count": len(topic_files),
        "topics_dir": str(topics_dir),
        "topic_files": topic_files,
    }
    dump_json(intermediate_dir / "run_summary.json", summary_payload)

    print(f"[synthesize] 阶段完成，共 {len(topic_files)} 个主题文件，目录：{topics_dir}")


# ── 标签合并工具 ──


def _build_grouping_prompt(tag_names: List[str], target_count: int) -> str:
    """构建 LLM 分组提示词：输入标签名列表，输出分组 JSON。"""
    names_text = "\n".join(f"- {n}" for n in tag_names)
    return (
        f"以下是 {len(tag_names)} 个网文写作知识标签名称：\n\n"
        f"{names_text}\n\n"
        f"请将它们分为约 {target_count} 个大分类，每个分类包含内容相关的标签。\n\n"
        f"分类原则：\n"
        f"1. 内容高度相关的标签归为同一类\n"
        f"2. 每个分类至少包含 2 个标签\n"
        f"3. 分类名应简洁概括（如「角色塑造与代入感」「剧情结构与节奏」）\n"
        f"4. 每个标签只能出现在一个分类中\n"
        f"5. 所有标签都必须被分配到某个分类\n\n"
        f"请直接输出 JSON，格式如下（不要输出其他内容）：\n"
        f'{{"分类名1": ["标签A", "标签B"], "分类名2": ["标签C", "标签D"]}}'
    )


def _resolve_tag_mapping(
    tag_names: List[str],
    merge_config: SynthesizeMergeConfig,
    stage: StageConfig,
    endpoint_states: List[EndpointState],
    runtime: RuntimeConfig,
) -> Dict[str, List[str]]:
    """解析标签 → 大分类映射。手动映射优先，否则走 LLM。"""
    if merge_config.mapping:
        print("[synthesize] 使用手动标签映射")
        return merge_config.mapping

    if not merge_config.use_llm:
        print("[synthesize] use_llm=false 且无手动映射，每个标签独立成类")
        return {name: [name] for name in tag_names}

    print("[synthesize] 使用 LLM 自动分组标签...")
    prompt = _build_grouping_prompt(tag_names, merge_config.target_categories)
    try:
        raw = call_llm_with_failover(
            stage=stage,
            endpoint_states=endpoint_states,
            system_prompt="你是一个分类专家。请严格按要求输出 JSON。",
            user_prompt=prompt,
            runtime=runtime,
        )
        mapping: Dict[str, List[str]] = _repair_json(raw)
        if not isinstance(mapping, dict):
            raise PipelineError("LLM 标签分组返回的不是 JSON 对象")
    except PipelineError as exc:
        print(f"[synthesize] LLM 标签分组失败: {exc}，降级为每标签独立成类")
        return {name: [name] for name in tag_names}

    # 限制分类数
    if len(mapping) > merge_config.target_categories * 2:
        print(
            f"[synthesize] 警告：LLM 返回 {len(mapping)} 个分类，"
            f"超过目标 {merge_config.target_categories} 的两倍"
        )

    return mapping


def _apply_tag_grouping(
    buckets: Dict[str, List[str]],
    mapping: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """按 mapping 合并 buckets。未映射标签归入「未分类」。"""
    # 构建 tag → category 反向索引（去重）
    tag_to_category: Dict[str, str] = {}
    for category, tags in mapping.items():
        seen: set[str] = set()
        for tag in tags:
            tag = tag.strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            if tag in tag_to_category:
                print(
                    f"[synthesize] 警告：标签 '{tag}' 重复映射"
                    f"（'{tag_to_category[tag]}' vs '{category}'），保留首次")
                continue
            tag_to_category[tag] = category

    # 检查映射中引用但不存在的标签
    all_bucket_tags = set(buckets.keys())
    for category, tags in mapping.items():
        for tag in tags:
            tag = tag.strip()
            if tag and tag not in all_bucket_tags:
                print(f"[synthesize] 警告：映射中标签 '{tag}' 不存在于实际标签集，跳过")

    merged: Dict[str, List[str]] = {}
    for tag_name, units in buckets.items():
        category = tag_to_category.get(tag_name, "未分类")
        merged.setdefault(category, []).extend(units)

    return merged



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


def run_pipeline(config: PipelineConfig, only_stage: str) -> None:
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
        run_synthesize_stage(config, extract_dir)


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
        run_pipeline(config, args.only_stage)
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
