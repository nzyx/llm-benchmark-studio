"""Benchmark parameter registry - the single source of truth.

Why this module exists
----------------------
Trying a first benchmark run, it is genuinely unclear what ``max_tokens``
means: it is the **output** cap, not the context length, not the input
length.  Rather than scattering hints across the UI, every parameter is
declared once here with its category, unit, bounds, the metrics it moves, and
a plain-language explanation.

The classification is the important part.  Once Context and Concurrency
benchmarks arrive, "how much do we feed the model" (input) and "how much do
we let it write" (output) must never be conflated again:

    model      - capabilities reported by the server, read-only
    input      - how much the model is asked to READ   -> drives Prefill / TTFT
    output     - how much the model is allowed to WRITE -> drives Decode / E2E
    sampling   - what the model writes
    execution  - how the harness runs (warmup, repeats, timeout, concurrency)

The frontend renders labels, units and tooltips straight from ``/api/params``
so the two sides can never drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import (
    DEFAULT_CONCURRENCY_LADDER,
    DEFAULT_CONTEXT_LADDER,
    HARD_MAX_CONCURRENCY,
    HARD_MAX_OUTPUT_TOKENS,
    HARD_MAX_REQUESTS,
    HARD_MAX_TIMEOUT_S,
)


# --------------------------------------------------------------------------
# categories
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    blurb: str
    order: int


CATEGORIES: list[Category] = [
    Category("model", "模型能力（只读）",
             "由服务端上报，用来判断上限。不可修改。", 0),
    Category("input", "输入侧 · Input",
             "决定「喂给模型读多少」。直接决定 Prefill 耗时与 TTFT。", 1),
    Category("output", "输出侧 · Output",
             "决定「让模型写多少」。直接决定 Decode 时长与端到端延迟。", 2),
    Category("sampling", "采样 · Sampling",
             "决定「模型写出什么内容」，不改变工作量级。", 3),
    Category("execution", "执行控制 · Execution",
             "决定「怎么跑、跑几遍、跑多久」，影响统计可信度。", 4),
]


# --------------------------------------------------------------------------
# parameter spec
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    category: str
    kind: str                          # int | float | bool | text | select | readonly
    default: Any = None
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    summary: str = ""                  # one-line label suffix / hint
    detail: str = ""                   # full explanation incl. misconceptions
    affects: list[str] = field(default_factory=list)
    introduced: str = "Phase 1"
    status: str = "active"             # active | planned | readonly

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------
#: Default minimum samples per concurrency level.  Kept as a named constant
#: because two places must agree on it: the parameter's own default, and the
#: "did the caller pin it?" test in `build_concurrency_config` (a value equal
#: to this one means "size it for me", any other value is a deliberate choice).
DEFAULT_REQUESTS_PER_LEVEL = 12

PARAMS: list[ParamSpec] = [
    # ---------------- model (read-only, filled from the probe) -------------
    ParamSpec(
        key="context_window", label="上下文窗口", category="model",
        kind="readonly", unit="tokens", status="readonly",
        summary="输入 + 输出的总容量上限",
        detail=(
            "模型单次请求能容纳的 **输入 tokens + 输出 tokens 之和** 的上限，"
            "由服务端上报（本机服务为 1,048,576）。\n\n"
            "常见误解：它不是 max_tokens，也不是「必须用满」的值。"
            "只有输入接近这个数字时，才会出现上下文压力。"
        ),
        affects=["能否运行", "prefill_tps", "TTFT"],
    ),
    ParamSpec(
        key="server_max_output", label="服务端输出硬上限", category="model",
        kind="readonly", unit="tokens", status="readonly",
        summary="服务端自己允许的 max_tokens 天花板",
        detail=(
            "服务端声明的 max_tokens 上限（本机为 65536）。"
            "即使前端填更大，服务端也会截断。"
        ),
        affects=["max_tokens"],
    ),
    ParamSpec(
        key="server_default_output", label="服务端默认输出", category="model",
        kind="readonly", unit="tokens", status="readonly",
        summary="不传 max_tokens 时服务端自己用的值",
        detail="服务端在请求未带 max_tokens 时采用的默认值（本机为 8192）。",
        affects=["max_tokens"],
    ),
    ParamSpec(
        key="server_slots", label="服务端并发槽位", category="model",
        kind="readonly", unit="路", status="readonly",
        summary="引擎一次能真正并行处理的请求数",
        detail=(
            "llama.cpp 系的 slot 数（来自 `/health.slots` 或 `/props.total_slots`）。"
            "它决定了并发测试的可扩展上限：并发数超过槽位数后，多出来的请求只能排队，"
            "此时延迟明显劣化而吞吐不再增长。\n\n"
            "vLLM 没有 slot 概念，对应的是启动参数 `--max-num-seqs`，HTTP 侧读不到。"
        ),
        affects=["并发吞吐", "排队延迟"],
        introduced="Phase 3",
    ),
    ParamSpec(
        key="server_kv_pool", label="KV pool 容量", category="model",
        kind="readonly", unit="tokens", status="readonly",
        summary="引擎可用于 KV cache 的总位置数",
        detail=(
            "引擎上报的 KV cache 总容量（tokens）。长上下文测试时它是真正的天花板："
            "`并发数 × 单请求上下文长度` 超过这个值就会开始排队、抢占或拒绝。\n\n"
            "来源：llama.cpp 系的 `/health`，或 vLLM 的 "
            "`vllm:cache_config_info{kv_cache_size_tokens}`。"
        ),
        affects=["最大可验证上下文", "并发上限"],
        introduced="Phase 3",
    ),
    ParamSpec(
        key="server_kv_concurrency", label="KV 容量推算并发上限",
        category="model", kind="readonly", unit="路", status="readonly",
        summary="按当前 KV 容量估算的同时可容纳请求数",
        detail=(
            "vLLM 的 `vllm:cache_config_info{kv_cache_max_concurrency}`，"
            "由 KV 池大小与单请求上下文长度推算。\n\n"
            "**这是 KV 容量给出的上限，不等于 `--max-num-seqs` 配置的调度上限**，"
            "两者取小才是真实并发天花板。并发测试超过它时会看到排队与抢占。"
        ),
        affects=["并发上限", "TTFT P95/P99"],
        introduced="Phase 7",
    ),

    # ---------------- input -------------------------------------------------
    ParamSpec(
        key="prompt", label="提示词", category="input", kind="text",
        summary="喂给模型的输入内容",
        detail=(
            "这就是**输入**。它的长度（按 token 计）决定了模型要读多少内容，"
            "从而决定 Prefill 耗时与 TTFT。\n\n"
            "注意：有 prompt cache 的服务端，重复发送同一段 prompt 会命中缓存，"
            "把 Prefill 测得虚高。做正式性能测试时应让每次采样的输入不重复。"
        ),
        affects=["input_tokens", "prefill_tps", "TTFT"],
    ),
    ParamSpec(
        key="use_chat", label="使用 Chat 模板", category="input", kind="bool",
        default=True,
        summary="走 /v1/chat/completions 而不是 /v1/completions",
        detail=(
            "开启后用 chat 模板（会额外包一层 role/结构），这部分也计入输入 tokens。"
            "关闭后走 /v1/completions，是纯粹的续写，没有思考链开销，"
            "适合测「干净的 Prefill」。"
        ),
        affects=["input_tokens", "TTFT"],
    ),
    ParamSpec(
        key="context_ladder", label="上下文长度阶梯", category="input",
        kind="text",
        default="512,1024,2048,4096,8192,16384,32768,65536,131072,262144",
        summary="要测的**输入**长度列表，逗号分隔；支持 512 / 1K / 256K 写法",
        detail=(
            "**这是输入长度，不是 max_tokens。** Context Benchmark 会逐档按这个"
            "长度构造输入，测出 Prefill / TTFT 随输入增长的变化曲线。\n\n"
            "写法：`512,1K,4K,32K,128K`，也接受纯数字。上限取"
            "「模型上下文窗口 − 输出预留 / 本工具硬上限 / 安全上限」三者最小值，"
            "超出的档位会被跳过并在结果里说明原因。\n\n"
            "长度是按**真实 token 数**构造的 —— 引擎提供 `/tokenize` 时用它精确"
            "测量，并用 `/detokenize` 裁剪到位；拿不到就退化为字符估算并明确标注。\n\n"
            "⚠️ **成本与风险**：TTFT 随输入近似线性增长。真机实测（Qwen3.8-Flash "
            "Q4_K_XL）512 输入 TTFT≈1.9s，8K 输入≈19.5s —— 一路跑到 256K 单档"
            "就要十几分钟，整条默认阶梯可能接近半小时。更危险的是**长 prefill "
            "可能直接把服务打挂**（实测在一台机器上 128K 输入让 llama.cpp 进程"
            "退出并重启）。建议先用短阶梯（如 `512,2K,8K`）确认量级，再逐步加长。"
        ),
        affects=["prefill_tps", "TTFT", "KV cache 占用"],
        introduced="Phase 5",
    ),
    ParamSpec(
        key="unique_prompt", label="每次采样使用不重复输入", category="input",
        kind="bool", default=True,
        summary="默认开启：彻底绕开 prompt 缓存（与 cache_prompt 互补）",
        detail=(
            "开启后每个样本的输入都会带一个**开头的**唯一标记，因此没有任何两次"
            "输入是相同的。\n\n"
            "标记必须放在**开头**：引擎缓存的是**前缀**，把标记加在末尾的话，"
            "前面整段正文仍然构成可命中的前缀 —— 实测在真机上仍会命中 "
            "62% 的缓存（`cache_n` = 64/103）。\n\n"
            "为什么还需要它：`cache_prompt=false` 依赖**引擎愿意配合**。"
            "遇到忽略该字段的服务端时，唯一输入是最后一道保险 —— 内容不同，"
            "缓存必然不命中。\n\n"
            "**默认开启**，因为同一个引擎缓存命中与否可以差 6.7 倍，而"
            "「开着更好看」。想特意测量缓存命中的场景再手动关掉 —— 关掉后"
            "结果页会明确警告哪些样本复用了同一输入。\n\n"
            "标记宽度固定，所以各样本输入长度一致。"
        ),
        affects=["prefill_tps", "TTFT"],
        introduced="Phase 4",
    ),
    ParamSpec(
        key="context_fill", label="长输入构造方式", category="input",
        kind="select", default="repeat_padding",
        choices=["repeat_padding", "corpus"],
        summary="怎么把输入撑到目标长度",
        detail=(
            "`repeat_padding`：重复同一段技术文本。简单、可预测，但短输入是长输入的"
            "**前缀** —— 这正是 Context 任务会给每档输入加唯一标记的原因。\n\n"
            "`corpus`：循环使用一小段多样语料，文本更自然，分布更接近真实负载。\n\n"
            "两种方式都会在开头插入唯一标记，**结构性杜绝**任何前缀缓存复用："
            "否则长档位会蹭到短档位的 KV cache，让长上下文看起来反而更快。"
        ),
        affects=["prefill_tps", "代表性"],
        introduced="Phase 5",
    ),
    ParamSpec(
        key="detect_max_context", label="自动探测最大上下文", category="execution",
        kind="bool", default=True,
        summary="二分搜索「实测能通过」的最大输入长度",
        detail=(
            "跑完阶梯后用**二分搜索**找出引擎实际能接受的输入上限，结果名为 "
            "**Maximum Verified Context** —— 是*本次实测通过*的长度，"
            "**不是模型的理论上下文上限**。\n\n"
            "安全约束：硬上限 + 每档独立 timeout + 只对传输类失败重试"
            "（被拒绝的输入重试没有意义）+ 步数封顶，不会无限循环。"
        ),
        affects=["可用上下文", "报告结论"],
        introduced="Phase 5",
    ),
    ParamSpec(
        key="input_tokens_actual", label="实际输入 tokens", category="input",
        kind="readonly", unit="tokens", status="readonly",
        summary="服务端回报的真实输入长度",
        detail="以服务端 usage.prompt_tokens 为准，包含 chat 模板的开销。做报告时以此为准，不要用字符数估算。",
        affects=["prefill_tps"],
    ),

    # ---------------- output ------------------------------------------------
    ParamSpec(
        key="max_tokens", label="输出上限 max_tokens", category="output",
        kind="int", unit="tokens", default=256, min=1,
        max=HARD_MAX_OUTPUT_TOKENS, step=64,
        summary="让模型「最多写多少」，不是上下文长度",
        detail=(
            "**这是输出侧参数：单次请求允许模型生成的最大 token 数。**\n\n"
            "它不等于上下文窗口，也不等于输入长度。上下文窗口是「输入 + 输出」"
            "的总容量上限，而 max_tokens 只掐住输出那一头。\n\n"
            "为什么它重要：输出越多，Decode 阶段越长，E2E 延迟基本线性增长，"
            "而 TTFT/Prefill 几乎不受影响。所以测吞吐和测延迟时要用不同的值。\n\n"
            "推理模型要注意：思考链（reasoning）token 也算在 max_tokens 里。"
            "设得太小，可能整段预算都被思考消耗掉，正文一个字都没写出来。"
        ),
        affects=["output_tokens", "decode_tps", "E2E", "TPOT"],
    ),
    ParamSpec(
        key="output_tokens_actual", label="实际输出 tokens", category="output",
        kind="readonly", unit="tokens", status="readonly",
        summary="服务端回报的真实输出长度",
        detail=(
            "以服务端 usage.completion_tokens 为准，**含思考链 token**。"
            "Decode 速度 = 该值 / (E2E − TTFT)。"
        ),
        affects=["decode_tps", "TPOT"],
    ),

    # ---------------- sampling ----------------------------------------------
    ParamSpec(
        key="temperature", label="温度 temperature", category="sampling",
        kind="float", default=0.0, min=0.0, max=2.0, step=0.1,
        summary="采样随机性；0 = 贪心解码",
        detail=(
            "0 表示贪心解码，输出确定、可复现，也是 benchmark 的默认值。\n"
            "大于 0 会引入随机性：内容更多样，但速度也会变化（本机服务在 "
            "temperature > 0 时切换到采样解码路径）。"
        ),
        affects=["输出内容", "decode_tps"],
    ),
    ParamSpec(
        key="seed", label="随机种子 seed", category="sampling",
        kind="int", default=None, min=0, max=4294967295,
        summary="固定采样结果，保证可复现",
        detail=(
            "与 `temperature > 0` 搭配使用。实测：同一 seed + 同一输入 → "
            "输出**逐字节相同**；换一个 seed 输出即改变。\n\n"
            "benchmark 默认 `temperature=0`（贪心解码），此时 seed 没有意义；"
            "只有做「随机采样下的稳定性 / 多样性」测试时才需要它。"
        ),
        affects=["可复现性"],
        introduced="Phase 4",
    ),
    ParamSpec(
        key="enable_thinking", label="启用思考链", category="sampling",
        kind="bool", default=None,
        summary="是否让模型先输出 reasoning；留空用服务端默认",
        detail=(
            "开启时 reasoning token 计入输出，会显著拉长 E2E；关闭后输出更「干净」，"
            "适合测纯 Decode 速度。\n\n"
            "实测（llama.cpp / Qwen3.8-Flash-Next）：关闭思考链后 reasoning 从 103 字降到 **0**，"
            "TTFT 从 696ms 降到 469ms。\n\n"
            "**发送方式很关键**：必须包进 `chat_template_kwargs.enable_thinking`。"
            "实测顶层直接传 `enable_thinking` 会被**静默忽略**（输出与默认完全一致）。"
        ),
        affects=["output_tokens", "E2E", "decode_tps"],
        introduced="Phase 4",
    ),

    # ---------------- execution ---------------------------------------------
    ParamSpec(
        key="warmup", label="预热次数", category="execution",
        kind="int", default=1, min=0, max=10,
        summary="不计入统计的预备请求",
        detail=(
            "第一次请求通常包含模型/权重加载、CUDA/ROCm 图编译、prompt cache 填充"
            "等一次性开销，直接计入会严重污染均值。预热请求的结果会被丢弃。"
        ),
        affects=["统计可信度"],
    ),
    ParamSpec(
        key="repeats", label="正式请求次数", category="execution",
        kind="int", default=3, min=1, max=HARD_MAX_REQUESTS,
        summary="用于统计的采样次数",
        detail=(
            "次数越多，P95/P99 越可信，但总耗时线性增长。\n"
            "经验值：快速看量级用 5 次；要报 P99 至少 20 次。"
        ),
        affects=["P95", "P99", "标准差"],
    ),
    ParamSpec(
        key="concurrency_ladder", label="并发阶梯", category="execution",
        kind="text",
        default="1,2,4,8,16,32",
        summary="要测的同时请求数列表，逗号分隔",
        detail=(
            "并发测的是「多用户同时用」时吞吐怎么涨、延迟怎么劣化。\n\n"
            "每一档会同时发出 N 个请求（N = 该档并发数），"
            "测完后汇总这一档的吞吐与延迟分布。\n\n"
            "**两个吞吐要分开看**：`Request Throughput`（每秒完成几个请求）"
            "通常先涨后平，`Token Throughput`（每秒产出多少 token）才反映真实算力天花板。\n\n"
            "超过服务端槽位数（`server_slots`）的档位仍然有意义 —— 请求会排队，"
            "正是要测排队时的延迟劣化；但结果是「排队后的表现」，"
            "不是「引擎能并行这么多」。"
        ),
        affects=["吞吐", "TTFT P95/P99", "错误率"],
        introduced="Phase 6",
    ),
    ParamSpec(
        key="requests_per_level", label="每档请求数", category="execution",
        kind="int", unit="个", default=DEFAULT_REQUESTS_PER_LEVEL, min=1,
        max=HARD_MAX_REQUESTS,
        summary="每个并发档位最少发多少个请求",
        detail=(
            "每档**最少**发几个请求。要报 P95/P99 就不能太小：\n"
            "- 12 个：看得出趋势，P95 勉强可用\n"
            "- 30 个以上：P99 才有意义\n\n"
            "**一档并发 N 要想真的测到 N 路并行，请求数得明显多于 N**："
            "请求是按波发的，每波起跑和收尾阶段在飞数都不满，"
            "所以只发 N 个时实测并发度只有 ~0.7N，两波才接近 0.94N。\n\n"
            f"* 保持默认值 {DEFAULT_REQUESTS_PER_LEVEL}：工具会自动按 "
            f"`max({DEFAULT_REQUESTS_PER_LEVEL}, 2×并发数)` 给每一档分配请求数"
            "（高并发档才多花时间，低档位不多测），并在参数警告里列出明细。\n"
            "* 填成**别的值**：完全按你填的数字发，一档就是这么多；"
            "低到测不动某些档位时会明确标出来，但不会擅自改你的设置。\n\n"
            "**总请求数 = 各档请求数之和**（按上面规则算，不是简单乘档位个数），"
            "受单次总量上限约束，超了会从最高档往下裁。"
        ),
        affects=["P95", "P99", "吞吐稳定性"],
        introduced="Phase 6",
    ),
    ParamSpec(
        key="cache_prompt", label="复用 prompt 缓存", category="execution",
        kind="bool", default=False,
        summary="默认关闭：开着测到的是缓存，不是算力",
        detail=(
            "开启时服务端复用上次相同前缀的 KV cache，**测量的就不再是引擎的 prefill "
            "能力，而是缓存命中**。\n\n"
            "实测证据（llama.cpp / Qwen3.8-Flash-Next，88 tokens 输入，同样 3 次采样）：\n"
            "· 开启 → TTFT **124 ms**，prefill 707 tok/s\n"
            "· 关闭 → TTFT **836 ms**，prefill 105 tok/s\n"
            "**6.7 倍差距**，而且方向是「开着更好看」。所以默认关闭。\n\n"
            "什么时候该开：想测「多轮对话 / 同前缀请求」的**稳态体验**时。"
            "但这时必须把它写进报告，否则数字会被误读 —— 本工具会把该配置标注在结果页。\n\n"
            "只有 llama.cpp 系引擎支持这个开关；不支持的引擎会收到一条 warning 并忽略。"
        ),
        affects=["prefill_tps", "TTFT"],
        introduced="Phase 3",
    ),
    ParamSpec(
        key="stream", label="流式请求", category="execution",
        kind="bool", default=True,
        summary="打开才能测出 TTFT / ITL",
        detail=(
            "关闭流式只能拿到总耗时，无法拆出 TTFT（首 token 延迟）和 ITL"
            "（token 间隔），Decode 速度只能靠总时长反推，精度差很多。\n"
            "Benchmark 默认开启；请求时同时带 stream_options.include_usage，"
            "以便服务端回传精确 token 数。"
        ),
        affects=["TTFT", "ITL", "decode_tps"],
    ),
    ParamSpec(
        key="timeout_s", label="单请求超时", category="execution",
        kind="float", unit="秒", default=120.0, min=1.0,
        max=HARD_MAX_TIMEOUT_S, step=10,
        summary="等多久算超时",
        detail=(
            "长上下文 + 大 max_tokens 时很容易超过默认值，建议按 "
            "`预计首 token 时间 + 输出 token 数 / 预期速度` 估算后再留 2 倍余量。"
        ),
        affects=["超时率"],
    ),
    ParamSpec(
        key="retries", label="重试次数", category="execution",
        kind="int", default=2, min=0, max=5,
        summary="失败后自动重试几次",
        detail=(
            "只对可重试的失败生效（连接错误、超时、429、5xx），4xx 不会重试。\n"
            "**流式请求一旦已经开始吐字就不再重试**，否则会重复计数 token。\n"
            "重试会体现在错误率统计里，报告会记录发生了什么。"
        ),
        affects=["错误率", "总耗时"],
    ),
]

PARAM_INDEX: dict[str, ParamSpec] = {p.key: p for p in PARAMS}


# --------------------------------------------------------------------------
# the relationship explainer (this is what actually fixed the confusion)
# --------------------------------------------------------------------------
RELATIONSHIP: dict[str, Any] = {
    "headline": "一次请求的 token 总量 = 输入 tokens + 输出 tokens，两者之和必须 ≤ 上下文窗口",
    "formula": "输入 tokens（受 context_ladder 控制）+ 输出 tokens（受 max_tokens 控制） ≤ 上下文窗口",
    "glossary": [
        {
            "term": "上下文窗口 context window",
            "side": "model",
            "text": "模型一次能同时「看到」的容量上限，输入和输出共用这个额度。本机服务是 1,048,576。",
        },
        {
            "term": "输入 tokens",
            "side": "input",
            "text": "你发给模型的内容有多长（含 chat 模板）。越长 Prefill 越久、TTFT 越高、KV cache 占用越大。Context Benchmark 就是在扫这个维度。",
        },
        {
            "term": "max_tokens",
            "side": "output",
            "text": "只限制模型「最多写多少」。写得多则 Decode 久、E2E 高，但对 TTFT 基本没影响。",
        },
        {
            "term": "实际输出 tokens",
            "side": "output",
            "text": "模型真正写了多少（含思考链）。它 ≤ max_tokens。计算 Decode 速度用的是这个值。",
        },
    ],
    "examples": [
        {
            "label": "短输入 + 短输出",
            "input_tokens": 88, "max_tokens": 128, "actual_output": 128,
            "note": "TTFT 低、E2E 短，适合测「响应速度」。",
        },
        {
            "label": "短输入 + 长输出",
            "input_tokens": 88, "max_tokens": 4096, "actual_output": 4096,
            "note": "TTFT 基本不变，E2E 线性拉长，适合测「Decode 吞吐」。",
        },
        {
            "label": "长输入 + 短输出",
            "input_tokens": 131072, "max_tokens": 128, "actual_output": 128,
            "note": "TTFT 急剧升高、Prefill 是瓶颈，适合测「长上下文能力」。",
        },
    ],
}


# --------------------------------------------------------------------------
# helpers used by the API layer
# --------------------------------------------------------------------------
def registry_payload() -> dict[str, Any]:
    """Everything the frontend needs to render the parameter UI."""
    return {
        "categories": [asdict(c) for c in CATEGORIES],
        "params": [p.to_dict() for p in PARAMS],
        "relationship": RELATIONSHIP,
        "limits": {
            "hard_max_output_tokens": HARD_MAX_OUTPUT_TOKENS,
            "hard_max_concurrency": HARD_MAX_CONCURRENCY,
            "hard_max_timeout_s": HARD_MAX_TIMEOUT_S,
            "hard_max_requests": HARD_MAX_REQUESTS,
        },
    }


def clamp_param(key: str, value: Any) -> tuple[Any, str | None]:
    """Clamp ``value`` for ``key``. Returns ``(value, warning_or_None)``."""
    spec = PARAM_INDEX.get(key)
    if spec is None:
        return value, f"未知参数 {key}，已忽略"
    if value is None:
        return spec.default, None
    try:
        if spec.kind == "int":
            num = int(value)
        elif spec.kind == "float":
            num = float(value)
        elif spec.kind == "bool":
            return bool(value), None
        else:
            return value, None
    except (TypeError, ValueError):
        return spec.default, f"{key} 不是合法数字，已回退到默认值 {spec.default}"

    original = num
    if spec.min is not None and num < spec.min:
        num = spec.min
    if spec.max is not None and num > spec.max:
        num = spec.max
    if num != original:
        return num, f"{key}={original} 超出允许范围 [{spec.min}, {spec.max}]，已收敛到 {num}"
    return num, None


def validate_settings(overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clamp every known key in ``overrides`` and report what was adjusted."""
    cfg: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in (overrides or {}).items():
        if key not in PARAM_INDEX:
            continue
        value, warn = clamp_param(key, value)
        cfg[key] = value
        if warn:
            warnings.append(warn)
    return cfg, warnings


def defaults_for(category: str) -> dict[str, Any]:
    return {
        p.key: p.default
        for p in PARAMS
        if p.category == category and p.status == "active"
    }


# Keys the performance task actually consumes today.
PERFORMANCE_KEYS: tuple[str, ...] = (
    "max_tokens", "warmup", "repeats", "temperature", "use_chat", "stream",
    "cache_prompt", "seed", "enable_thinking", "unique_prompt",
)
# Keys that are passthrough text rather than registry parameters.
PASSTHROUGH_KEYS: tuple[str, ...] = ("prompt", "messages", "options")
# Registry parameters forwarded verbatim into the request body as top-level
# fields (llama.cpp / OpenAI both accept these directly).
ENGINE_OPTION_KEYS: tuple[str, ...] = ("cache_prompt", "seed")
# `enable_thinking` is NOT forwarded top-level: measured against llama.cpp,
# a top-level `enable_thinking` is silently ignored, while
# `chat_template_kwargs.enable_thinking` works.  See build_performance_config.
NESTED_OPTION_KEYS: tuple[str, ...] = ("enable_thinking",)


def build_performance_config(
    model_id: str,
    overrides: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Turn client overrides into a clamped, warning-annotated task config.

    Unknown keys and parameters that belong to a later phase are reported
    instead of being silently dropped - the UI surfaces those warnings.
    """
    overrides = dict(overrides or {})
    warnings: list[str] = []

    for key in overrides:
        if (key in PASSTHROUGH_KEYS or key in PERFORMANCE_KEYS
                or key in ENGINE_OPTION_KEYS or key in NESTED_OPTION_KEYS):
            continue
        spec = PARAM_INDEX.get(key)
        if spec is None:
            warnings.append(f"未知参数 `{key}`，已忽略")
        elif spec.status == "planned":
            warnings.append(
                f"`{key}` 计划在 {spec.introduced} 生效，当前版本不会影响结果"
            )
        elif spec.status == "readonly":
            warnings.append(f"`{key}` 是服务端只读值，不能由客户端设置，已忽略")
        else:
            warnings.append(f"`{key}` 与性能测试无关，已忽略")

    cfg: dict[str, Any] = {
        key: PARAM_INDEX[key].default for key in PERFORMANCE_KEYS
    }
    valid, clamp_warnings = validate_settings({
        k: v for k, v in overrides.items()
        if k in PERFORMANCE_KEYS or k in ENGINE_OPTION_KEYS
        or k in NESTED_OPTION_KEYS
    })
    cfg.update(valid)
    warnings.extend(clamp_warnings)

    cfg["model"] = model_id
    cfg["prompt"] = overrides.get("prompt")
    cfg["messages"] = overrides.get("messages")

    # engine-specific options are merged into ``options`` which the adapter
    # spreads into the request body
    options = dict(overrides.get("options") or {})
    for key in ENGINE_OPTION_KEYS:
        if cfg.get(key) is not None:
            options[key] = cfg[key]

    # `enable_thinking` must be nested; measured against llama.cpp, the
    # top-level form is silently ignored.
    thinking = cfg.get("enable_thinking")
    if thinking is not None:
        nested = dict(options.get("chat_template_kwargs") or {})
        nested["enable_thinking"] = bool(thinking)
        options["chat_template_kwargs"] = nested

    cfg["options"] = options or None
    return cfg, warnings


# --------------------------------------------------------------------------
# context task
# --------------------------------------------------------------------------
CONTEXT_KEYS: tuple[str, ...] = (
    "max_tokens", "warmup", "repeats", "temperature", "use_chat",
    "context_ladder", "context_fill", "detect_max_context",
    "cache_prompt", "unique_prompt", "seed", "enable_thinking",
)
# Keys every task understands; served as the "common" group in the UI.
COMMON_KEYS: tuple[str, ...] = (
    "max_tokens", "temperature", "warmup", "repeats", "timeout_s", "retries",
)

_LADDER_SUFFIX = {"k": 1024, "m": 1024 * 1024}


def foreign_param_warnings(overrides: dict[str, Any],
                           allowed: tuple[str, ...],
                           task_label: str) -> list[str]:
    """Report every parameter this task is going to ignore.

    "Known parameter, wrong task" is the quietest kind of failure: the request
    succeeds, the run looks configured, and the knob simply does nothing.
    A parameter the registry has never heard of at least deserves the same
    treatment.
    """
    out: list[str] = []
    for key in overrides:
        if key in PASSTHROUGH_KEYS or key in allowed:
            continue
        if key in PARAM_INDEX:
            out.append(f"`{key}` 与{task_label}无关，已忽略")
        else:
            out.append(f"未知参数 `{key}`，已忽略")
    return out


def parse_ladder(text: str | None,
                 label: str = "context_ladder") -> tuple[list[int], list[str]]:
    """Parse ``"512,1K,4M"`` into sorted unique counts.

    Shared by the context ladder (token counts) and the concurrency ladder
    (parallel request counts) - both are "a list of numbers the user typed".
    """
    warnings: list[str] = []
    if not text or not str(text).strip():
        return [], [f"{label} 为空，使用默认阶梯"]
    out: set[int] = set()
    for raw in str(text).replace("，", ",").split(","):
        item = raw.strip().lower().replace("_", "")
        if not item:
            continue
        mult = 1
        if item and item[-1] in _LADDER_SUFFIX:
            mult = _LADDER_SUFFIX[item[-1]]
            item = item[:-1]
        try:
            value = int(float(item) * mult)
        except ValueError:
            warnings.append(f"无法解析 {label} 中的 `{raw.strip()}`，已忽略")
            continue
        if value <= 0:
            warnings.append(f"{label} 含非法值 `{raw.strip()}`，已忽略")
            continue
        out.add(value)
    if not out:
        return [], [f"{label} 没有可用值，使用默认阶梯"]
    return sorted(out), warnings


def build_context_config(
    model_id: str,
    overrides: dict[str, Any] | None,
    *,
    server_context: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Clamp the Context task config the same way the performance task does."""
    overrides = dict(overrides or {})
    warnings: list[str] = []

    warnings.extend(foreign_param_warnings(overrides, CONTEXT_KEYS, "上下文测试"))

    cfg: dict[str, Any] = {
        key: PARAM_INDEX[key].default for key in CONTEXT_KEYS
    }
    valid, clamp_warnings = validate_settings(
        {k: v for k, v in overrides.items() if k in CONTEXT_KEYS})
    cfg.update(valid)
    warnings.extend(clamp_warnings)

    ladder, ladder_warnings = parse_ladder(cfg.get("context_ladder"))
    warnings.extend(ladder_warnings)
    if not ladder:
        ladder = list(DEFAULT_CONTEXT_LADDER)
    cfg["context_ladder"] = ladder

    cfg["model"] = model_id
    cfg["server_context"] = server_context
    cfg["prompt"] = overrides.get("prompt")
    cfg["messages"] = overrides.get("messages")

    options = dict(overrides.get("options") or {})
    for key in ENGINE_OPTION_KEYS:
        if cfg.get(key) is not None:
            options[key] = cfg[key]
    thinking = cfg.get("enable_thinking")
    if thinking is not None:
        nested = dict(options.get("chat_template_kwargs") or {})
        nested["enable_thinking"] = bool(thinking)
        options["chat_template_kwargs"] = nested
    cfg["options"] = options or None
    return cfg, warnings


# --------------------------------------------------------------------------
# concurrency task
# --------------------------------------------------------------------------
CONCURRENCY_KEYS: tuple[str, ...] = (
    "max_tokens", "warmup", "temperature", "use_chat",
    "concurrency_ladder", "requests_per_level",
    "cache_prompt", "unique_prompt", "seed", "enable_thinking",
)


def build_concurrency_config(
    model_id: str,
    overrides: dict[str, Any] | None,
    *,
    server_slots: int | None = None,
    server_context: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Clamp the Concurrency task config the same way the other tasks do."""
    overrides = dict(overrides or {})
    warnings: list[str] = []

    warnings.extend(foreign_param_warnings(overrides, CONCURRENCY_KEYS,
                                           "并发测试"))

    cfg: dict[str, Any] = {
        key: PARAM_INDEX[key].default for key in CONCURRENCY_KEYS
    }
    valid, clamp_warnings = validate_settings(
        {k: v for k, v in overrides.items() if k in CONCURRENCY_KEYS})
    cfg.update(valid)
    warnings.extend(clamp_warnings)

    ladder, ladder_warnings = parse_ladder(cfg.get("concurrency_ladder"),
                                           label="concurrency_ladder")
    warnings.extend(ladder_warnings)
    if not ladder:
        ladder = list(DEFAULT_CONCURRENCY_LADDER)

    # The hard cap is not a suggestion - a run that asks for more concurrency
    # than we allow would put the engine (and the machine) at risk.
    over_cap = [c for c in ladder if c > HARD_MAX_CONCURRENCY]
    if over_cap:
        ladder = [c for c in ladder if c <= HARD_MAX_CONCURRENCY]
        warnings.append(
            f"并发档位 {over_cap} 超过硬上限 {HARD_MAX_CONCURRENCY}，已移除"
        )
    if not ladder:
        ladder = list(DEFAULT_CONCURRENCY_LADDER)
        warnings.append("所有并发档位都超限，已回退到默认阶梯")
    cfg["concurrency_ladder"] = ladder

    # A level of N can only *demonstrate* N-way concurrency if it fires well
    # more than N requests.  The harness fires a level's requests in waves, so
    # the in-flight count climbs at the start of every wave and drains at the
    # end: measured on a real vLLM node, level 16 given 32 requests (2 waves)
    # read eff 14.99 (= 0.94*N), while level 32 given 32 requests (1 wave) read
    # eff 22.69 (= 0.71*N).  One wave under-reports its own level; two are
    # enough - and two waves are *cheaper* than a flat count for low levels.
    #
    # Which remedy is right depends on whether the caller actually picked the
    # number.  Presence cannot answer that: the UI posts every parameter
    # pre-filled with the registry default, so `requests_per_level` is always
    # "present" - including when nobody touched it.  A value that differs from
    # the registry default is a deliberate choice and is honoured exactly;
    # the default value itself means "size it for me".
    floor = max(1, int(cfg.get("requests_per_level") or 1))
    default_floor = int(PARAM_INDEX["requests_per_level"].default or 1)
    top = max(ladder) if ladder else 1
    if floor != default_floor:
        cfg["requests_per_level_mode"] = "flat"
        if floor < top:
            unmeasurable = [c for c in ladder if c > floor]
            warnings.append(
                f"按你设定的每档请求数 {floor}，档位 {unmeasurable} 的在飞请求数"
                f"最多只有 {floor} —— 它们测的不是自己声称的并发能力，"
                "已在结果里标为「采样不足」。工具不会擅自改动你的设置；"
                "想测得动这些档位，就把每档请求数提到 ≥ 2×并发档"
            )
        elif floor < top * 2:
            warnings.append(
                f"每档请求数 {floor} 只比最高并发档 {top} 多出不到一倍："
                "每档在飞请求数是先升后降的（起跑和收尾阶段都不满），"
                "并发越高这个影响越明显，吞吐会被稀释。"
                f"要稳定的并发读数建议 ≥ {top * 2}，"
                f"或者把并发阶梯收到 {top // 2} 以内"
            )
    else:
        cfg["requests_per_level_mode"] = "waves"
        dropped: list[int] = []
        while True:
            counts = [max(floor, 2 * c) for c in ladder]
            total = sum(counts)
            if total <= HARD_MAX_REQUESTS or len(ladder) <= 1:
                break
            dropped.append(ladder.pop())
        cfg["concurrency_ladder"] = ladder
        if dropped:
            warnings.append(
                f"按「每档至少 2×并发数」取样，总请求数会超过单次上限 "
                f"{HARD_MAX_REQUESTS}，已移除最高档 {dropped}；"
                f"实际每档请求数 {counts}，总请求数 {total}"
            )
        if top and max(counts) > floor:
            warnings.append(
            f"每档请求数按「至少 2×并发数」自动确定（你设的 {floor} 作为下限）："
            f"{dict(zip(ladder, counts))}，总请求数 {total}。"
            "一档只发 N 个请求时，均值在飞数会被头尾稀释到 ~0.7N，"
            "两波才有 0.85–0.95N —— 这是让每一档都能自证并发度的最小代价"
            )

    # Above the engine's own slot count, requests queue instead of running in
    # parallel.  Still worth measuring - that is exactly the "many users at
    # once" case - but the report must not read it as parallel capacity.
    if server_slots and max(ladder) > server_slots:
        cfg["queues_above_slots"] = True
        warnings.append(
            f"最高档 {max(ladder)} 超过服务端槽位数 {server_slots} —— "
            "超出部分的请求会排队，测到的是排队后的延迟，不是并行能力"
        )

    cfg["model"] = model_id
    cfg["server_slots"] = server_slots
    cfg["server_context"] = server_context
    cfg["prompt"] = overrides.get("prompt")
    cfg["messages"] = overrides.get("messages")

    options = dict(overrides.get("options") or {})
    for key in ENGINE_OPTION_KEYS:
        if cfg.get(key) is not None:
            options[key] = cfg[key]
    thinking = cfg.get("enable_thinking")
    if thinking is not None:
        nested = dict(options.get("chat_template_kwargs") or {})
        nested["enable_thinking"] = bool(thinking)
        options["chat_template_kwargs"] = nested
    cfg["options"] = options or None
    return cfg, warnings
