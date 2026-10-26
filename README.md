# Local LLM Benchmark Studio

面向本地部署大模型 HTTP 服务的可视化 Benchmark 工作台。
输入一个 OpenAI-compatible 地址（llama.cpp / vLLM / halogen / LM Studio …），
自动探测服务、跑性能测试、记录结果、生成对比。

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)

- 前端：React + TypeScript + Vite
- 后端：Python + FastAPI
- 压测：asyncio + httpx（真实流式计时）
- 存储：SQLite
- 图表：ECharts
- 部署：Docker Compose（后续阶段）

> 直接上手：见 [快速开始](#快速开始) —— 两条命令，后端 8000 / 前端 5173。

## 为什么不是「又一个压测脚本」

大部分同类工具给出数字，但不告诉你**这个数字能不能信**。本地推理的测量坑极多，
而且几乎每一个都朝「结果更好看」的方向偏。这个项目把**测量有效性**当成一等功能：

| 陷阱 | 会把结果歪成什么 | 这里的处理 |
|---|---|---|
| 命中 prompt cache | prefill 虚高 **6.7 倍** | 每样本唯一输入（标记在**开头**）+ 用引擎自报的 `cache_n` 交叉校验 |
| 输出被合并成一次 flush | decode 报出 **196,078 tok/s**（真实 ~135） | 首末 token 事件跨度不足时**拒绝出数**并说明原因 |
| 排队等待被算进 prefill | 单个档位虚低 **42%** | 用引擎自报 `prompt_ms` 对照，失败后先等引擎排空再继续 |
| 并发档位 > 每档请求数 | 曲线「见顶」其实是采样不足 | 逐档输出可达并发度，不足即标记，且绝不擅自改用户设的采样数 |
| 客户端测的并发 ≠ 引擎真并行 | 「32 路并发」其实只跑了 16 路 | 在负载进行中采样引擎自己的 `requests_running` |
| 把测量误差当成引擎特性 | 得出改了也没用的优化建议 | 用引擎计数器增量（抢占/排队/成功率）判定归因，无证据就**明确否掉** |

这些结论都附带真机实测数据，见
[三个任务的口径必须一致](#三个任务的口径必须一致这是被真实数据逼出来的一节) 与各任务的实测小节。

---

## 当前进度

| Phase | 内容 | 状态 |
|---|---|---|
| 1 | 项目骨架 + 最小可运行闭环 | ✅ 完成 |
| 2 | API Client 抽离 + **参数体系** | ✅ 完成 |
| 3 | llama.cpp / vLLM Adapter + **引擎指纹识别** | ✅ 完成 |
| 4 | Performance Benchmark + **测量有效性控制** | ✅ 完成 |
| 5 | Context Benchmark + **Maximum Verified Context** | ✅ 完成 |
| 6 | Concurrency Benchmark + **实测并发度** | ✅ 完成 |
| 7 | Quality Benchmark | ⏳ |
| 8 | GPU Host Agent | ⏳ |
| 9 | Dashboard | ⏳ |
| 10 | HTML Report | ⏳ |
| 11 | Docker | ⏳ |
| 12 | End-to-End Test | ⏳ |

**Phase 1 已打通的闭环**：
输入 HTTP API → 探测并获取 Model → 健康检查 → 发一次真实流式请求 →
测出 TTFT / Prefill / Decode / TPOT / ITL / E2E → 落 SQLite →
前端展示 + 历史列表 + 实时进度（SSE）。

**Phase 2 已完成的**：
把传输层从 Adapter 里抽成独立的 `ApiClient`（连接池 / 超时 / 重试 /
流式行迭代 / Key 脱敏），并建立**参数注册表**（`app/params.py`）作为
所有参数的单一事实源，前端通过 `/api/params` 渲染标签、单位、范围与说明。

**Phase 3 已完成的**：
三层 Adapter 结构 + 基于**可观测证据**的引擎指纹识别，以及把 llama.cpp 与
vLLM 完全不同的指标名归一到同一套 schema。

**Phase 5 已完成的**：
Context Benchmark（性能随输入长度变化）+ **Maximum Verified Context** 自动探测，
按**精确 token 数**构造输入，能区分「引擎拒绝」与「引擎被打挂」。

**Phase 6 已完成的**：
Concurrency Benchmark（吞吐与延迟随并发变化），吞吐按**整档墙钟时间**计算，
并报告 **实测并发度**（Little's law）来证明请求真的同时在飞。

---

## Concurrency Benchmark（Phase 6）

### 吞吐必须按墙钟时间算，不能把单请求耗时相加

把每个请求的耗时加起来再"求吞吐"，会为**实际上一个接一个跑**的请求报出漂亮的
吞吐数字。这里每一档的吞吐都来自整个 batch 的墙钟时间，而 batch 内部用
`asyncio.Semaphore(并发数)` 真正并发发出。

### 实测并发度：判断这张表可不可信的那个数

```
实测并发度 = Σ(每个成功请求的 e2e) ÷ 该档墙钟时间
```

这是 Little's law 的直接应用。如果「并发 8」这一档的实测并发度只有 ≈1，
说明请求其实被串行化了，这一档的吞吐不能读作并发能力。现在这个值会在
表格里显示，低于 1.3 时标红，也会写成一条数据警告。

mock 上实测：并发 1 → **1.00**，并发 4 → **3.97**，并发 8 → **7.91**。
这三条断言就是"harness 没在偷偷串行"的证明。

### 超过引擎槽位数的档位：排队 ≠ 并行

引擎自己会报槽位数（`/health` 的 `slots`，这里是 4）。超过它的档位仍然要测
—— 那正是「多用户同时用」的真实场景 —— 但报告必须说清楚：

- 配置阶段给出 warning（"超出部分的请求会排队"）
- 表格里那些行标上「排队」
- 结果里带 `queues_above_slots: true`

### ⚠️ 每档请求数：一档要能"自证"并发度（这里连着犯过两个错）

```
一档的实测并发度 ≤ min(并发档位, 在飞请求数的均值)
```

只有 12 个请求，就不可能有 128 个请求同时在飞。默认配置曾经是
**阶梯到 32、每档 12 个请求** —— 也就是说**开箱配置就自带一半假档位**。

真机实测（某 vLLM 节点，阶梯 `1..128`、每档 12 个请求）：

| 档位 | 请求数 | 实测并发度 | 上限 |
|---|---|---|---|
| 8 | 12 | 6.09 | 8 |
| 16 | 12 | 10.34 | **12** |
| 32 | 12 | 8.91 | **12** |
| 64 | 12 | 11.17 | **12** |
| 128 | 12 | 11.07 | **12** |

16/32/64/128 四档测到的其实是**同一个并发水平**（≈11），
于是曲线"见顶"了 —— 那个顶不是引擎的能力，是请求数不够。

**第一个错**：当时的修法是「自动把每档请求数提到不低于最高并发档」。
它悄悄改写了用户显式设定的采样数 —— 用户设 3 想快速看一眼，被顶到 32
（192 个请求，**10.7 倍代价**：同一条阶梯从 ~8s 变成 82s，`max_tokens=256`
那次跑了 **350s**）。用户的原话是「变慢了非常多」。

**第二个错**：档位只发 3 个请求时（采样不足），报告却信心十足地宣布
「实测并发度在 8 档到达 2.91 并停止增长 —— **这就是这台机器的真实并发天花板**」。
3 个样本推不出硬件上限。

现在的规则：

1. **用户填的值就是用户的值**。`requests_per_level` 保持默认（12）时，工具按
   `max(12, 2×并发数)` 给每一档分配（低档位不多测，只有高并发档才多花时间），
   并在警告里列出明细；填成**别的值**就完全按这个值发，绝不改写。
   判定按**值**而不是「参数是否出现」—— 因为前端会把所有参数连同默认值一起提交，
   「出现过」永远为真（这个坑下面单独说）。
2. 一档"采样不足"有**两种**原因，分别标注、分别措辞：
   - **请求数不够**（`请求数 < 档位`）：在飞数不可能超过请求数；
   - **只发了一波**（`请求数 == 档位`）：请求数够，但每波起跑/收尾阶段在飞数不满。
     实测**一波只有 ~0.7N，两波才有 0.85–0.95N**。
3. 结果里带 `achievable_concurrency`、`underprovisioned`、`headline_levels`；
   表格多一列「上限」，采样不足的行打斜纹标 †，并说明是哪一种原因。
   **头条指标（peak / at_concurrency）只在能自证并发度的档位里取** ——
   "并发 32"档如果实际只在飞 22 个，它不该提供被引用的峰值。
4. **采样不足的表不产生"硬件天花板"结论**：`_batching_profile()` 在
   可用档位少于 2 个、或任一档样本数 < 8 时直接返回空 ——
   与其给一个自信的错误结论，不如不给。

同一台机器、同一条阶梯（`1..32`，`max_tokens=64`）：

| | 换引擎修好前 | 现在 |
|---|---|---|
| 总请求数 | 192 | **148** |
| 耗时 | 81.8s | **45.3s** |
| 32 档的 eff/N | 0.71 | **0.84** |
| 有多少档被判"采样不足" | 0（但实际都测不准） | 0（**且真的都测准了**） |

### ⚠️ 前端把所有参数连同默认值一起提交，把预设"顶掉"了

`App.tsx` 的表单原先用注册表默认值初始化，然后**把全部参数发出去**
（`pick(values, TASK_KEYS)`）。后果有两个：

* `requests_per_level` 永远是"出现过"的，后端没法区分"用户填的 12"和"预填的 12"；
* **各预设的调优值从来没生效过** —— 并发预设写的是 `max_tokens=128`，
  实际每次跑的都是注册表默认 `256`，**每个请求贵一倍**（350s 那次就是这么来的）。

现在表单是三层叠加：`注册表默认 → 所选预设的调优值 → 用户改过的键`，
界面上显示的就是真正会跑的值（`touched` 只记录用户动过的键）。

### 吞吐曲线非线性：先分清两种原因

一条"不线性"的吞吐曲线有两种完全不同的成因，而补救方法相反：

| 现象 | 原因 | 怎么办 |
|---|---|---|
| 实测并发度 ≪ 档位 | 请求没被真正重叠 | 加请求数 / 查引擎并发上限 |
| 实测并发度 ≈ 档位，但单请求 TPOT 暴涨 | **引擎的批处理代价** | 改引擎配置或换机器，加并发没用 |

第二种是真实且常见的：**batch-1 走快路径**（投机解码常常只在 batch 1 划算）。
真机实测同一台 vLLM：

| 并发 | 单请求 TPOT | 单请求 decode | 总 Token 吞吐 |
|---|---|---|---|
| 1 | **5.25 ms** | 190.4 tok/s | 132.8 |
| 2 | **23.34 ms** | ~43 tok/s | **76.2** ← 反而腰斩 |
| 4 | 26.14 ms | ~38 tok/s | 128.9 |
| 16 | 27.43 ms | ~36 tok/s | 463.1 |
| 64 | 27.67 ms | ~36 tok/s | 479.0 |

**1→2 单请求慢 4.4 倍、总吞吐腰斩，到并发 ~16 才回升到基线以上。**
这是引擎特性，不是测量误差 —— 而且是我们**在 harness 之外独立复现**的：
直接发 6 个串行请求 + 3 组两两并发，得到 190 → 72 tok/s，同一形状。

报告现在会：
- 从 `metrics_snapshot` 的 before/after 取引擎计数器的**增量**
  （`preemptions_total` / `requests_waiting` / `requests_success_total`），
  作为"是否过载"的证据；
- 有证据才说"服务端拒请求或排队"（并建议查 `--max-num-seqs`）；
- 没证据就**明确否掉**这个归因，改为点名批处理代价与实测并发天花板；
- 结果里带 `result.batching`（代价倍数、峰值、天花板）和
  `result.engine_evidence`，前端在曲线下方给一条「并发为什么没换来吞吐」。

⚠️ 旧的「吞吐在 1→2 处明显下降，通常意味着服务端开始拒请求或排队超时」
**是误诊**：实测那台引擎 `preemptions_total` 增量为 0、`requests_waiting`
始终为 0、成功率 100%。归因错了会把排查引向 `--max-num-seqs`，而真正的原因
是 batch-1 快路径，改那个参数没用。

### 两个问题必须分开：**并行度** vs **容量**

`effective_concurrency`（Σ 单请求耗时 ÷ 墙钟）数的是「**在飞**」，不是「**在跑**」——
请求在引擎内部排队的时间也算进去了。所以它会显得远高于引擎真正的并行度，
而一个客户端**无法**从自己的计时里分辨「引擎同时跑了 32 个」和「跑了 16 个、另 16 个在排队」。

唯一独立见证是引擎自己的仪表，而且必须在**负载进行中**采样
（前后快照看到的都是空闲引擎）。真机实测（vLLM `127.0.0.1:8080`，
`--max-num-seqs=16`，阶梯 1/8/32/64）：

| 并发 | 请求数 | 实测并发度 | **引擎在跑（均值/峰值）** | 引擎排队峰值 | Token 吞吐 |
|---|---|---|---|---|---|
| 1 | 12 | 1.0 | 0.61 / 1 | 0 | 133 |
| 8 | 16 | 7.6 | 6.5 / 8 | 0 | 240 |
| 32 | 64 | 27.2 | 14.2 / **16** | 16 | **468** |
| 64 | 128 | 50.9 | 14.8 / **16** | 48 | **476** |

**引擎同时在跑的从来没超过 16**，而客户端算出 50.9 —— 差额全是排队。
于是报告把两个问题分开回答：

- **并行度** → 「引擎在跑」列（引擎自报），上限约 **16**（vLLM 的 `--max-num-seqs`）。
  并发 > 16 的档位测到的是「排队后的延迟」，不是它们声称的并发能力。
- **容量** → 「峰值 Token 吞吐」= **所有档位的最高值**（含排队档位）。
  排队不是测量误差 —— **排队正是批处理保持满载、把吞吐压到顶的方式**，
  所以 468/476 tok/s 才是这台机器的容量。

⚠️ 这里我犯过一个错，值得记下来：最初把「引擎排队」的档位从峰值里剔除了，
于是报出 **234.7 tok/s 的「峰值」**，而实际容量是 460+ —— **把机器低估了一半**。
「并行度」和「容量」的样本集必须分开选，不能都只取「能自证并发」的那些档位。
（另一个同源的坑：`parallel_peak` 的兜底逻辑会把排队档位放回来，而我们有**直接证据**
它没并行 —— 兜底因此是错的，宁可返回空。）

引擎饱和后就该停止加并发。报告现在直接给一句平台结论：

> 并发 ≥ 16 的档位（引擎已饱和）聚合吞吐基本持平：并发 32 档 468、
> 并发 64 档 476 tok/s（相差 2%）—— 这台机器的**容量就是这个水平**，
> 再多加并发只会增加排队和延迟，不会提高吞吐

### 真机实测（halogen fork / halogen-qwen3.8-flash-next，每档 4 个请求）

| 并发 | 墙钟 | 请求吞吐 | Token 吞吐 | TTFT p50 / p95 |
|---|---|---|---|---|
| 1 | 4.64 s | 0.86 req/s | 27.6 tok/s | 387 / 531 ms |
| 2 | 3.48 s | 1.15 req/s | 36.8 tok/s | 581 / 785 ms |
| 4 | 2.98 s | 1.34 req/s | 43.0 tok/s | 1040 / 1564 ms |

**这张表本身就是结论**：并发从 1 提到 4，Token 吞吐只涨 **1.56×**，而 TTFT p95
劣化 **2.9×**（531 → 1564 ms）。也就是说这台机器**并发买不到吞吐，只买到延迟** ——
典型的 decode 阶段受内存带宽限制、设备已经跑满的特征。
如果只看「请求吞吐从 0.86 涨到 1.34」会得出相反的印象，所以两个吞吐必须分开看。

---

## Context Benchmark（Phase 5）

### 输入是"精确 token 数"，不是字符估算

引擎提供 `/tokenize` 时用它测量、用 `/detokenize` 裁剪到位，**实际 token 数**
与目标值一起记录（`built.actual_tokens` / `built.source`），报告里两者分开显示。
真机实测偏差 **0.00%**；拿不到 tokenizer 会退化为字符估算并明确标 `估算`。

### 每档输入都唯一：结构性绕开缓存

阶梯测试的关键陷阱是：短输入是长输入的**字面前缀**。带 prompt cache 的引擎会把
长档位的 prefill 一部分从缓存里端出来，于是**长上下文反而更快**。

所以每档输入的第一个 token 都是一个唯一标记（`[ctxprobe <8位随机>]`），
**不管引擎理不理解 `cache_prompt`，前缀缓存都不可能命中**。
实测：真机 TTFT 随输入单调增长（512→1.9s，8K→19.5s），没有出现倒挂。

### Maximum Verified Context：叫法本身就是结论的一部分

自动探测用**二分搜索**，结果的字段名就叫 `max_verified_context`，
值只在**本次运行实测通过**的意义上成立，`disclaimer` 里明确拒绝把它说成理论上限。

三个设计点：

1. **阈值用阶梯本身夹逼**。跑完阶梯后就已知 `[最大通过, 首次失败]`，
   二分只在夹逼区间内跑，不再从 0 到上限重探 —— 真机/mock 实测
   只需 **3 步**就收敛，而且**永远不会向上探到已知会失败的档位**
   （真机上 128K 那次就是这样把服务打挂的）。
2. **输出预算从可用上千里扣掉**：输入和输出共用同一个上下文窗口，
   可用输入上限 = `min(引擎窗口, 硬上限, 安全上限) − max_tokens`。
   最后一档等于整窗口的配置会被直接跳过并说明原因，而不是去撞一次失败。
3. **失联就中止，绝不编数**。见下。

### 失败分类：引擎拒绝 vs 引擎被打挂

这两件事长得像（都是请求失败），含义完全相反。`app/errors.py::classify_error`
把每个失败样本分成 `too_long / unreachable / timeout / auth / server_error / unknown`：

| 现象 | 分类 | 处理 |
|---|---|---|
| HTTP 400/413（引擎活着，明确说不行） | `too_long` | 真实容量边界，阶梯停止向上 |
| `Connection refused`（服务不在了） | `unreachable` | **中止运行**，结果标 `partial` |
| `RemoteProtocolError: peer closed connection` | `too_long` | 视为容量边界（但会顺手做存活检查） |
| 读超时 | `timeout` | 重试，重试完仍失败才算 |

任何一档失败后都会做一次**便宜的存活探测**（`GET /v1/models`，5s 超时，
走 `provider.ping()`），并给引擎 3×5s 的**恢复窗口** —— 因为它可能是 OOM 重启，
几秒后自己就回来了。仍未恢复 → 整个运行中止：

- 运行状态是 **`partial`**（不是 `completed`，也不是 `failed`）
- 未尝试的档位**逐条列出**，标 `engine_unavailable`，不静默丢弃
- `max_verified_context.aborted = true`，`value` 只是"中断前实测通过的最大值"，
  `upper_bound` 是"首次观察到的失败值"，并提示重跑

### 真机实测（llama.cpp / Qwen3.8-Flash-Next-UD-Q4_K_XL，单档 1 次采样）

| 目标输入 | 实际 tokens | TTFT | Prefill | 说明 |
|---|---|---|---|---|
| 512 | 512 | 1,865 ms | 308 tok/s | |
| 2,048 | 2,048 | 5,325 ms | 396 tok/s | |
| 8,192 | 8,246 | 19,544 ms | 425 tok/s | |
| 16,384 | 16,492 | 40,399 ms | 410 tok/s | |
| 32,768 | 32,851 | 90,243 ms | 365 tok/s | |
| 65,536 | — | 失败 | — | 见下方「超时按工作量放宽」 |

拟合：**TTFT ≈ 0.53 s 固定开销 + 2.2 ms/token**（边际 ~450 tok/s），
32K 起轻微超线性（attention 变长），decode 随 KV 变长从 25.7 降到 20.6 tok/s。
每档 3 次采样偏差 ±0.3% —— **这些数字是这台机器的真实算力，不是测量误差**。
TTFT 随输入长度本来就是秒到分钟级；觉得"太慢"时先确认缓存是关的，
再确认自己在测的是长输入。

### 单请求超时按工作量放宽（`scaled_timeout`）

固定超时会**在引擎还在算的时候掐断请求**：64K 的 prefill 按 365 tok/s 需要
~180s，而默认超时是 120s —— 实测那条请求正好在 120.08s 处断掉，
随后引擎失联。掐断本身可能就是把引擎带走的原因。

所以 Context 任务的每个请求超时 = `max(用户配置, 估算耗时)`，其中
估算 = `输入 tokens ÷ 150 + 输出 tokens ÷ 10 + 30s`（保守假设），
仍受 `HARD_MAX_TIMEOUT_S=1800` 硬封顶；显式配置了更长的超时则不会被缩短。
放宽会在进度流和结果警告里说明（例如"64K 档超时已放宽到 474s"）。

另有一条数据质量警告：**输出预算可能被思考链整个吃掉**（所有输出 token 都是
reasoning、正文为 0、`ttft_content` 缺失），此时会建议上下文测试改用
`enable_thinking=false`。

### 成本提示

TTFT 随输入**近似线性**，所以长阶梯很贵：单跑一条 512→256K 的默认阶梯
可能接近半小时，并且**长 prefill 有把服务打挂的实测先例**
（那次请求把 llama.cpp 进程打退出，之后所有请求变成 connection refused）。
建议先用短阶梯（`512,2K,8K`）确认量级。

---

## 引擎适配与指纹识别

```
OpenAICompatProvider        通用：/v1/models、/v1/chat/completions、/metrics
    ├── LlamaCppProvider    + /props、/slots、/tokenize、llamacpp:* 指标
    └── VLLMProvider        + 空 200 /health、vllm:* 指标、/tokenize
```

检测只依据**观测到的证据**，并把证据一并回报（前端「判定依据」里可见），
不猜、不掩盖：

| 信号 | 结论 |
|---|---|
| `/metrics` 含 `vllm:*` | vLLM（置信度 0.95） |
| `/metrics` 含 `llamacpp:*` | llama.cpp 系（0.95） |
| `/props` 或 `/slots` 有响应 | llama.cpp 系（0.85） |
| `/health` 暴露 slots / kv_pool 字段 | llama.cpp 系 fork（0.70） |
| `/health` 返回空 200 | vLLM（0.60） |
| 都识别不出 | 通用适配器（0.30），并明确说明无法识别 |

**关键点：每个原生探测都是可选的。** 缺少 `/props`、`/slots` 的 fork
仍会被正确判为 llama.cpp 系，只是能力清单里这几项标 `×`。
另外用了一个小技巧：对 POST-only 路由发 GET，**405 就证明路由存在**，
这样探测 `/completion`、`/tokenize` 不需要消耗任何生成 token。

### 指标归一化

两个引擎的指标名完全不同，报告层不该关心是谁产出的：

| 概念 | llama.cpp | vLLM |
|---|---|---|
| 运行中请求 | `llamacpp:requests_processing` | `vllm:num_requests_running` |
| 排队请求 | `llamacpp:requests_deferred` | `vllm:num_requests_waiting` |
| KV cache 占用 | `llamacpp:kv_cache_usage_ratio` | `vllm:gpu_cache_usage_perc` |
| 生成 tokens | `llamacpp:tokens_predicted_total` | `vllm:generation_tokens_total` |
| TTFT | 不暴露 | `vllm:time_to_first_token_seconds_*` |

统一输出 `normalized` 字段，并派生 `prompt_tokens_per_s`、
`generation_tokens_per_s`、`draft_accept_ratio`。
vLLM 的指标带 label，`Prometheus` 解析器会按名字**求和**折叠
（`request_success_total{finished_reason=...}` 求和后才是真实总数）。

### 新参数

| key | 分类 | 说明 |
|---|---|---|
| `server_slots` | model（只读） | 并发槽位数，决定并发测试的可扩展上限 |
| `server_kv_pool` | model（只读） | KV pool 位置数，长上下文测试的真天花板 |
| `cache_prompt` | execution | **默认 false**；开着测到的是缓存不是算力。仅 llama.cpp 系支持，不支持会给 warning |
| `unique_prompt` | input | 每次采样输入内容不重复，**内容层面**彻底绕开缓存（与 `cache_prompt` 互补） |
| `seed` | sampling | 固定采样结果；实测同 seed 输出逐字节相同 |
| `enable_thinking` | sampling | 三态（服务端默认 / 开 / 关）；**必须包在 `chat_template_kwargs` 里** |
| `context_ladder` | input | 要测的**输入**长度阶梯，支持 `512,1K,4K,32K,128K` 写法 |
| `context_fill` | input | 长输入怎么造：`repeat_padding`（重复文本）/ `corpus`（多样语料） |
| `detect_max_context` | execution | 二分搜索 Maximum Verified Context；受硬上限/超时/步数三重约束 |
| `concurrency_ladder` | execution | 要测的并发阶梯（默认 `1,2,4,8,16,32`），超硬上限的档位会被移除 |
| `requests_per_level` | execution | 每个并发档位发几个请求（默认 12；要报 P99 至少 30） |

### Phase 4 实测（真机 llama.cpp / Qwen3.8-Flash-Next）

| 场景 | TTFT | 结论 |
|---|---|---|
| 默认（`cache_prompt=false`） | **837 ms** | 冷 prefill，诚实数字 |
| `cache_prompt=true` | **158 ms** | 缓存在干活 |
| `cache_prompt=true` + `unique_prompt=true` | **557 ms** | 内容唯一 → 缓存必然不命中 |

思考链控制：`enable_thinking=true` → reasoning **681 字符**；`false` → **0**。

**两个"字段发对了没有"的坑（都是实测出来的）：**
1. `enable_thinking` 放在**顶层会被静默忽略**，必须包进
   `chat_template_kwargs.enable_thinking`。顶层形式与默认输出完全一致，很难察觉。
2. **不要相信引擎自报的 `reasoning_tokens`**：该引擎的流式 usage 里根本没有这个字段，
   自报值恒为 0。现在改为测量**实际收到的 reasoning 字符数**（`reasoning_chars`），
   自报值只作为回退。

### ⚠️ prompt cache 实测：6.7 倍差距（这是本项目最重要的一条方法论）

在真机（llama.cpp / Qwen3.8-Flash-Next，88 tokens 输入，同样 3 次采样）上 A/B：

| `cache_prompt` | TTFT | Prefill |
|---|---|---|
| `true`（复用缓存） | **124.5 ms** | 707.8 tok/s |
| `false`（每次重新 prefill） | **835.8 ms** | 105.3 tok/s |

结论：
1. 该引擎**确实理会** `cache_prompt`，即使走的是 `/v1/chat/completions`。
2. 复用缓存时数字**好看 6.7 倍**，但那是缓存在干活，不是引擎算力。
   冷启动的真实 prefill 只有 **~105 tok/s**。
3. 因此 `cache_prompt` 默认值定为 **false**，并且**运行配置会随结果一起展示**，
   开启时结果页会显示「⚠ 本次开启 prompt cache，TTFT / Prefill 偏乐观」。
4. Phase 1/2 期间记录的「TTFT ~14ms、prefill 数千 tok/s」全部是缓存污染的结果。

---

## 参数体系（重点：别再把「输入长度」和「输出上限」搞混）

所有参数按**作用位置**分五类，由 `backend/app/params.py` 统一定义，
前端 UI 与 `/api/params` 都从这一份定义渲染，不会出现两边说法不一致。

| 分类 | 中文名 | 决定什么 | 影响哪些指标 | 代表参数 |
|---|---|---|---|---|
| `model` | 模型能力（只读） | 服务端上报的上限 | 能否运行 | `context_window`、`server_max_output` |
| `input` | 输入侧 | 喂给模型**读多少** | Prefill、TTFT、KV 占用 | `prompt`、`use_chat`、`context_ladder`、`context_fill` |
| `output` | 输出侧 | 让模型**写多少** | Decode、TPOT、E2E | `max_tokens` |
| `sampling` | 采样 | 写出**什么内容** | 输出内容、可复现性 | `temperature`、`seed`、`enable_thinking` |
| `execution` | 执行控制 | **怎么跑、跑几遍** | P95/P99、错误率 | `warmup`、`repeats`、`cache_prompt`、`detect_max_context`、`concurrency_ladder`、`requests_per_level`、`timeout_s`、`retries` |

### 三者的关系（最容易混的地方）

```
输入 tokens（受 context_ladder 控制）
      +  输出 tokens（受 max_tokens 控制）
      ≤  上下文窗口 context window（模型能力，只读）
```

Phase 5 起这条式子是**强制执行**的：Context 任务的可用输入上限 =
`min(引擎窗口, 硬上限, 安全上限) − max_tokens`，超出的档位会被跳过并说明原因。

- **上下文窗口** = 输入 + 输出的**总容量上限**，不是 `max_tokens`，也不是「必须用满」的值。
- **`max_tokens`** = 只掐输出那一头：让模型「最多写多少」。写得多 → Decode 更久、E2E 更高，但 **TTFT / Prefill 几乎不受影响**。
- **推理模型注意**：思考链（reasoning）token 也计入 `max_tokens`，设小了可能出现「预算被思考用光、正文一个字没写」。

参数状态分三档，前端会明确标注：

| status | 含义 |
|---|---|
| `active` | 当前版本生效，可直接调 |
| `planned` | 已在注册表登记，标注了接入阶段（P4/P5/P6），当前不影响结果 |
| `readonly` | 服务端上报，客户端不可改 |

服务端会**强制收敛**越界参数并回报 `settings_warnings`，例如
`max_tokens=999999` → 收敛到 65536；提交 Phase 5/6 的参数会提示「计划在 Phase X 生效」。

---

## 目录结构

```
llm-bench-test/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口、运行注册表、SSE 进度
│   │   ├── client.py            # ApiClient：连接池/超时/重试/流式行/脱敏
│   │   ├── errors.py            # ApiError（客户端与 Adapter 共用）
│   │   ├── params.py            # ★ 参数注册表：分类/单位/范围/说明/收敛
│   │   ├── prometheus.py        # Prometheus 文本解析（label、直方图、转义）
│   │   ├── metrics_normalize.py # ★ 引擎指标 → 统一 schema
│   │   ├── config.py            # 硬安全上限、presets、代理开关
│   │   ├── db.py                # SQLite 持久化
│   │   ├── metrics.py           # mean/median/p50/p95/p99
│   │   ├── schemas.py           # 请求模型
│   │   ├── providers/
│   │   │   ├── base.py          # Provider 统一接口（内置 ApiClient）
│   │   │   ├── openai_compat.py # 通用 OpenAI 兼容
│   │   │   ├── llamacpp.py      # + /props /slots /tokenize llamacpp:* 指标
│   │   │   ├── vllm.py          # + 空 200 /health、vllm:* 指标、/tokenize
│   │   │   └── detect.py        # ★ 指纹识别 + 适配器选择
│   │   └── engine/
│   │       ├── base.py          # BenchmarkTask 生命周期 + 统一结果 Schema + 存活探测
│   │       ├── probe.py         # ★ 单次流式测量 + 失败分类 + 缓存证据（各任务共用）
│   │       ├── performance.py   # PerformanceTask
│   │       ├── context_builder.py # ★ 按精确 token 数构造长输入（含探针标定）
│   │       ├── context.py       # ★ ContextTask + Maximum Verified Context
│   │       └── concurrency.py   # ★ ConcurrencyTask + 实测并发度
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 页面（按输入/输出/采样/执行分组渲染参数）
│   │   ├── api.ts / types.ts / format.ts
│   │   ├── render-check.tsx     # 无头渲染校验（npm run check:render）
│   │   └── components/
│   │       ├── ParamField.tsx      # 单参数控件 + ⓘ tooltip
│   │       ├── ParamGuide.tsx      # 参数速查面板
│   │       ├── ServiceDetails.tsx  # 引擎家族/适配器/能力/判定依据
│   │       ├── ConfigStrip.tsx     # 结果随运行配置一起展示
│   │       ├── ScaleChart.tsx      # ECharts 封装（SSR 安全）
│   │       ├── ContextView.tsx     # ★ Context Scaling 视图 + MVC 卡片
│   │       ├── ConcurrencyView.tsx # ★ 并发视图（双吞吐 + 分位延迟 + 实测并发度）
│   │       ├── RichText.tsx
│   │       └── ResultView.tsx
│   └── package.json
├── tools/
│   ├── smoke_test.py            # 端到端冒烟测试（需要真实引擎）
│   ├── phase3_check.py          # 识别 + 归一化 + cache/唯一输入/采样选项/vLLM
│   ├── consistency_check.py     # ★ 三个任务的口径一致性 + decode 可测性（自带 4 个 mock）
│   ├── phase5_check.py          # 阶梯/夹逼/二分/引擎失联中止（自带 mock，可 --real）
│   ├── phase6_check.py          # 并发吞吐/分位/实测并发度/失联中止（自带 mock，可 --real）
│   ├── real_engine_check.py     # 真机测量有效性校验（地址变更后必跑）
│   └── mock_engine.py           # 可配置假引擎：前缀缓存/思考字段/拒绝/崩溃/
│                                #   事件粒度(--tokens-per-chunk)/一次 flush(--coalesce)/
│                                #   抛弃后仍占槽(--abandon-*/--serialize)
└── data/bench.sqlite3           # 运行记录（自动创建）
```

---

## 快速开始

### 后端（端口 8000）

```powershell
$venv = "C:\Users\liuzihan\.workbuddy\binaries\python\envs\llm-bench\Scripts\python.exe"
cd backend
& $venv -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 前端（端口 5173，已配置 /api 代理）

```powershell
cd frontend
npm install
npm run dev
```

打开 http://localhost:5173 → 填 API 地址 → 探测服务 → Run Full Benchmark。
参数不懂就点右上角 **「参数速查」**，或把鼠标放在任意参数旁的 ⓘ 上。

### 测试

```powershell
# 端到端冒烟测试（前端 / 后端 / 代理 / 参数注册表 / 探测 / 识别 / 跑一次 / 历史）
python tools/smoke_test.py --llm http://127.0.0.1:8080

# 引擎识别 + 指标归一化 + cache / 唯一输入 / 采样选项 / vLLM 专项
#（自带 5 个 mock，无需真机）
python tools/phase3_check.py

# 三个任务的口径一致性：逐样本输入独立性、warmup 排除、prefill 公式、
# 目标长度 vs 实到长度（自带 2 个 mock，无需真机）
python tools/consistency_check.py

# Context 阶梯 / 夹逼 / 二分 / 引擎失联中止（自带 mock；加 --real <url> 再跑真机）
python tools/phase5_check.py
python tools/phase5_check.py --real http://127.0.0.1:8080

# 并发吞吐 / 分位延迟 / 实测并发度 / 失联中止（自带 mock，可加 --real）
python tools/phase6_check.py
python tools/phase6_check.py --real http://127.0.0.1:8080 --real-ladder 1,2,4

# 真机测量有效性校验：默认是否冷 prefill、cache 开关是否生效、
# unique_prompt 是否绕开缓存、enable_thinking 是否真的关掉思考链
python tools/real_engine_check.py --llm http://127.0.0.1:8080

# 无头渲染校验（浏览器跑不起来时的替代方案）
cd frontend; npm run check:render
```

单独启动某一种假引擎（调试适配器用）：

```powershell
python tools/mock_engine.py --flavor llamacpp-fork --port 8731
python tools/mock_engine.py --flavor vllm          --port 8732
python tools/mock_engine.py --flavor llamacpp-vanilla --port 8733

# 超长输入两种行为（Phase 5 用）：
python tools/mock_engine.py --port 8734 --context 65536 --max-context 12000  # 拒绝
python tools/mock_engine.py --port 8735 --context 65536 --die-above 12000    # 崩掉
# 并发用：跑到第 N 个请求就把服务打挂
python tools/mock_engine.py --port 8736 --die-after 6
# 没有 tokenizer 的引擎（真机 halogen fork 就是这样）
python tools/mock_engine.py --port 8737 --no-tokenize
# vLLM 的两个特性：思考链字段叫 `reasoning`，且 prefix caching 关不掉
python tools/mock_engine.py --flavor vllm --port 8738 \
    --thinking-field reasoning --thinking-tokens 24 --prefix-caching
# 流事件的粒度：真机 vLLM 开投机解码时一个事件带 ~2.3 个 token
python tools/mock_engine.py --flavor vllm --port 8739 --tokens-per-chunk 3
# 短输出被合并成一次 flush（真机 max_tokens=8：4 个事件挤在 0.036 ms 内到达）
python tools/mock_engine.py --flavor vllm --port 8740 --coalesce

# 大请求被中途抛弃、但引擎仍在算（真机 128K warmup 就是这样）。
# --serialize 让后续请求真的排队 —— 否则 mock 的多线程会把"忙"变成纯装饰
python tools/mock_engine.py --flavor llamacpp-fork --port 8741 \
    --prefill-seconds 6 --abandon-after 0.5 --abandon-above 4096 --serialize
# 同上但没有 /metrics：客户端无法排空，只能靠检测报警
python tools/mock_engine.py --flavor generic --port 8742 \
    --prefill-seconds 6 --abandon-after 0.5 --abandon-above 4096 --serialize
```

`--abandon-times`（默认 1）控制抛弃几次，`--abandon-above`（默认 0）只抛弃超过
该长度的请求 —— 两个都需要，否则那唯一一次抛弃会被长度标定的探针用掉。
`MOCK_DEBUG=1` 会把 mock 自己的判定（缓存命中、是否抛弃）写到 stderr。

---

## 统一结果 Schema（v1.0）

所有 Benchmark 结果共用一套 JSON 结构，字段见
`backend/app/engine/base.py::build_result_document`：

```
schema_version / run_id / created_at / task_type / preset / label / status
provider{kind,engine,base_url,api_version}
model{id,max_model_len,context_length,meta}
config / started_at / finished_at / duration_ms
samples[]        # 每次请求的原始指标
stats{}          # 每个指标的 count/mean/median/p50/p95/p99/min/max/std
                 # （Context 任务里是每个长度一个点的系列；
                 #    Concurrency 任务里是每个并发档一个点的系列）
result{}         # 任务级聚合
contexts[]       # Context 任务：每档输入的构造结果 + 指标 + 失败分类
levels[]         # Concurrency 任务：每个并发档的墙钟/吞吐/分位/错误
task_results{}   # 一个 run 跑多任务时，每个任务一个子文档
errors{count,rate,items[]}
metrics_snapshot{before,after}   # 来自服务端 /metrics
host_agent                       # Phase 8
```

`status` 除了 `completed / failed / timeout / cancelled` 还有 **`partial`**：
测到了真实数据，但没跑完请求的全部内容（目前只有一种情况 —— 引擎在长上下文
探测过程中失联）。把它叫 `completed` 是撒谎，叫 `failed` 又浪费了已经测到的数据。

### 每个失败样本带稳定分类

`samples[].error_kind` / `contexts[].failure_kind` 取值：
`too_long | unreachable | timeout | auth | server_error | unknown`。
它决定报告怎么写：`too_long` 是**真实的容量边界**，`unreachable` 是
**测量作废**，两者绝不能混。详见 `backend/app/errors.py::classify_error`。

### 指标定义（写在代码注释里，避免含糊）

| 指标 | 定义 |
|---|---|
| `ttft_ms` | 首个生成 token 的延迟（**含 reasoning token**） |
| `ttft_content_ms` | 首个**可见正文** token 的延迟 |
| `decode_window_ms` | `e2e_ms - ttft_ms`（整个请求的尾巴，含收尾开销） |
| `stream_events` | 实际携带 token 的流事件个数 |
| `decode_span_ms` | 首末 token 之间的时间差 —— **`tpot_ms` 的真正依据** |
| `tpot_ms` | `decode_span_ms / (output_tokens - 1)` |
| `decode_tps` | `1000 / tpot_ms` |
| `prefill_tps` | `input_tokens / ttft`（TTFT ≈ prefill 时间） |
| `queue_wait_ms` | `ttft_ms - server_prompt_ms`，TTFT 里**不是 prefill** 的那部分 |
| `itl_ms` | 相邻 chunk 到达间隔的均值（真实流式） |

`tpot_ms` / `decode_tps` 在无法测准时**置空**（`None`），原因写在
`decode_unmeasurable` 字段里 —— 见下一节。一个说不清来源的数字比没有数字更糟，
因为它会静默混进均值。

---

## 解码速度什么时候测不出来（"decode 数据不稳定"的真正原因）

**现象**：同一台 vLLM 节点，`decode_tps` 一列里出现 `23071`、`40864`、`53516`
甚至 `196078`。而它的真实解码速度只有 130 上下。同一档位内三次采样的比值
可以到 **169 倍**。

**根因**：`decode_tps` 的分母来自"token 什么时候到"。当引擎把**整个输出压缩成
一次 flush** 送达时，首末 token 的时间差塌缩到微秒级，而 `usage` 仍然老老实实
报 8 / 32 个 token。旧公式于是拿 0.036 ms 去除 7，得到 196078 tok/s —— 一个
物理上不可能的数，然后它进了报告均值。

真机实测（vLLM 0.29.0.dev0 / Qwen3.8-Flash-Next，投机解码开启）：

| `max_tokens` | 流事件数 | `usage` 输出 | 首末 token 跨度 | 旧公式给出 |
|---|---|---|---|---|
| 8 | 4 | 8 | **0.036 ms** | **196,078 tok/s** ❌ |
| 8 | 3 | 8 | **0.080 ms** | 87,610 tok/s ❌ |
| 16 | 9 | 16 | 72.7 ms | 206 tok/s |
| 32 | 15 | 32 | 229.9 ms | 134.9 tok/s |
| 64 | 29 | 64 | 443.9 ms | 141.9 tok/s |
| 128 | 55 | 128 | 833.9 ms | 152.3 tok/s |
| 256 | 61 | 141 | 936.2 ms | 149.5 tok/s |

**≥32 起就稳定在 135–152 tok/s** —— 那才是引擎的真实速度。短输出之所以会塌缩，
是因为投机解码一个调度步就能产出 ~2 tokens，几个步就走完 8 个 token，服务端
把这几步的输出合并成一次刷新送出。

**处理方式**（`probe.py`）：

1. `decode_span_ms` 用**首末 token 事件**的时间差，不再用 `e2e_ms - ttft_ms`；
2. 要求至少 2 个 token 事件，且跨度 ≥ `MIN_STREAM_GAP_MS × (事件数 - 1)`
   （1 ms/事件是客户端**计时分辨率**的下限，不是对引擎速度的判断）；
3. 不满足就**置空** `tpot_ms` / `decode_tps`，写入 `decode_unmeasurable`
   （`single_flush` / `coalesced_flush`），并由 `decode_coalesced_warning()`
   报警，三任务共用；
4. TTFT、token 计数、E2E 一切照旧 —— 只有测不准的那一项被扣下。

**运行前就会提示**：`max_tokens < 32` 时给一条 settings warning，
并在引擎开了投机解码时说明原因。

**注意**：`decode_tps` 的**分母永远是 `usage` 的 token 数，不是流事件数**。
真实引擎一个事件可以带 2–3 个 token（实测 `tok/event ≈ 2.0–2.3`），
按事件数算会低 2–3 倍。回归测试里专门有一个 3 tokens/事件 的 mock 对照。

### 投机解码必须上报

vLLM 从不在 `/health` 里说自己开了投机解码，唯一的痕迹是
`vllm:spec_decode_*` 计数器。读它有两个理由：

* 一个调度步产出**可变**的 token 数（草稿接受多少就是多少），所以 tok/s 天然
  比单 token 引擎抖 —— 不说清楚，用户会以为自己的机器有问题；
* 客户端若假设"一个流事件 = 一个 token"，速率和 TPOT 都会算错。

实测该节点：每步草拟 3.0 tokens、**接受率 30%**、实际每步产出 1.91 tokens。

---

## Prefill 为什么会"上下文越长越快"（以及那个假数据点）

**现象**：长上下文表格里，`Prefill` 从 2K 的 1304.9 tok/s 一路升到 16K 的
1666.2，而 128K 掉到 1083.0 之后又回升到 1426.8 —— 两头都不符合物理。

**两个独立的原因，都用真实数据定位：**

### 1. `prefill_tps` 是"平均值"，天然会把固定开销摊薄掉

`prefill_tps = input_tokens / ttft`，而 `ttft = 固定开销 + tokens / 真实速率`。
短档位里固定开销占比大，于是**平均速率**偏低；档位变长，固定开销被摊薄，
平均速率就"上升"——读起来像"上下文越长越快"，但这句话对一个吞吐指标是荒谬的。

真机实测（1.3M 上下文节点，用**引擎自报**的 `prompt_ms` 计算）：

| 目标 | 实际 tokens | 引擎 prefill 时间 | 引擎速率 | 客户端速率 |
|---|---|---|---|---|
| 2K | 2,050 | 1,551.8 ms | 1,321.0 | 1,304.9 |
| 8K | 8,202 | 5,308.7 ms | 1,545.0 | 1,537.0 |
| 16K | 16,400 | 9,759.8 ms | **1,680.4** ← 峰值 | 1,666.2 |
| 32K | 32,803 | 19,798.8 ms | 1,656.8 | 1,598.0 |
| 64K | 65,601 | 40,569.3 ms | 1,617.0 | 1,611.5 |
| 256K | 262,401 | 183,270.7 ms | 1,431.8 | 1,426.8 |
| 512K | 524,798 | 396,118.8 ms | 1,324.8 | 1,320.5 |

**16K 之前上升、之后单调下降** —— 这才是正确的形状：短档位在摊销固定开销，
长档位是真实的带宽退化（边际速率从 ~1,635 掉到 ~1,232 tok/s）。

现在报告里会给出**固定开销拟合**（`result.prefill_fit`）：

> 固定开销拟合：**0.30s** / 请求 + **1621 tok/s** 边际速率（4 个干净档位）。
> 短档位的 `prefill_tps` 偏低是因为固定开销被摊薄得少，不代表长档位真的更快。

拟合**只使用干净档位** —— 被污染的点会扭曲线斜率，那正是下面这条。

### 2. 排队等待被算进了 prefill（那个 1083 的假点）

128K 那一档：客户端 TTFT **121,144 ms**，引擎自报 prefill **85,410 ms**，
差 **35,734 ms（占 41.8%）**。其余各档这个差只有 0.3–3.7%。

原因是一次请求失败留下的"幽灵"：128K 的 **warmup 在读错误中死掉**
（`ReadError`，跑了 50.2 s、输出 0 字节），但引擎**不会因为客户端走了就停止
prefill** —— 它还需要约 85 s。我们紧接着发了正式请求，于是它只能排在那个
被抛弃的请求后面。

* 引擎自报的速率 **1536 tok/s** 完美落在趋势线上；
* 我们报的 **1083** 用的是一个 121 s 的 TTFT。

**三道防线**（都在共享代码里）：

1. **检测**：`queue_wait_ms = ttft_ms - server_prompt_ms`，超过
   `max(1s, 15% × TTFT)` 就置 `prefill_contaminated`，该档的 Prefill 列显示
   `n/a‡` 并给出悬浮说明。没有引擎自报时间就**不做任何断言**。
2. **排空**：任何样本失败后，先问引擎"还忙吗"（`llamacpp:requests_processing`
   / `vllm:num_requests_running`），等它空下来再发下一个请求。看不到该指标时
   退化为有界的短等待（诚实承认这是猜的）。
3. **报出来**：三个任务共用同一套告警，写明是哪个档、多少毫秒、占 TTFT 多少。

顺带发现的一个真实交互：**失败后重试的样本会命中 prompt cache。**
`client.py` 对"未产出任何字节"的传输错误会重试，而重试发的是**同一段文本** ——
有缓存的引擎直接命中（mock 实测 `cache_n=8224`，占输入 100%）。这正是
prompt cache 陷阱经由重试重新出现，现有的 `cache_warning` 会把它报出来。

---

## 三个任务的口径必须一致（这是被真实数据逼出来的一节）

Dashboard 把 Performance / Context / Concurrency 三份结果并排放，同名指标
必须是同一个东西。定义写在 `app/engine/probe.py` 的模块 docstring 里，
**聚合规则也写在共享代码里而不是各任务里**，否则一定会漂移。

### 三条强制规则

1. **同一个数字背后，样本必须是独立的。**
   每个样本记录 `input_hash`（实际发送内容的 sha256 前 12 位）。
   有 prompt cache 的引擎缓存的是**前缀**，所以"相同输入"意味着第二个样本
   大部分 prefill 被跳过。实测在同一档内：`prefill` 9289 → 61297 tok/s
   （**6.6 倍**）、TTFT 222ms → 34ms。这种均值既不是冷启动也不是稳态，
   所以 `input_identity_warning()` 会把它报出来，而不是默默平均。

2. **warmup 样本绝不进入统计。** 由 `summarize_samples()` 统一剔除，
   并回报 `warmup_excluded` 计数。历史 bug：Context 任务把 warmup 一起算进
   均值，而另外两个任务都排除了 —— 同一台机器、同一档位，三个任务的
   "均值"含义不同。

3. **每个样本的输入必须唯一**，且标记放在**开头**（前缀缓存）。
   `unique_prompt` 三个任务都默认开启；关掉时会报警。

### 三个任务的差异（这些是刻意的，不是口径不一致）

| | Performance | Context | Concurrency |
|---|---|---|---|
| 比较单位 | 整次运行 | 每个上下文档位 | 每个并发档位 |
| 每个单位几个样本 | `repeats` | `repeats`（warmup 另计） | `requests_per_level` |
| 输入长度 | 固定短输入 | 每档按**精确 token 数**构造 | 固定短输入 |
| `prefill_tps` | ✅ 均值 | ✅ 均值 | ❌ 不展示（并发下 prefill 与排队混在一起，展示会误读） |
| TTFT 展示 | 均值 | 均值 + p95 | **p50 / p95 / p99** |
| 吞吐 | 无（单请求） | 无 | **按整档墙钟时间算** |

`decode_tps` 在三处定义相同，但**配置不同**：preset 默认
`max_tokens` 分别是 256 / 16 / 128，而思考链也算进输出 token，
所以开了思考链时 context 的 decode 测的是 reasoning 速度 —— 三处都会
给出对应的 data warning。

### 回归测试

```powershell
python tools/consistency_check.py    # 三个任务的口径一致性（自带 2 个 mock）
```

覆盖四个层：

- **A 单元**：共享 helper 的行为（含 `decode_coalesced_warning` 的两种
  `decode_unmeasurable` 分支、以及"被扣下的样本不拖均值的后腿"）
- **B 端到端**：三个任务跑在同一台"缓存关不掉"的引擎上、
  `prefill == input_tokens / ttft` 逐样本校验、目标长度 vs 实到长度
- **C 对照**：缓存关不掉时唯一输入被强制覆盖并说明；缓存可关时用户选择被
  尊重且复用输入会被报警
- **D decode 可测性**：`--coalesce` 的假引擎下**必须不出数**并报警；
  `--tokens-per-chunk 3` 的假引擎下**必须出正确的数**（实测 51.8 vs 引擎
  真实 50 tok/s —— 按事件数算会低 3 倍）
- **E prefill 可测性**：引擎抛弃大请求后仍在算时 ——（① 有忙信号）harness 必须
  先排空再继续，该档不许被污染；（② 没有 `/metrics`）必须把排队等待**标记出来**
  并报警。实测 `queue_wait_ms=[5648.0, 29.1]`，被污染那档 `prefill_tps=1032.6`
  而引擎自报 `1366.7` —— 和真机的 1083 vs 1536 同构
- **F 并发口径**（`phase6_check.py` 的 A/D/F 层）：
  - 档位超过每档请求数时 `requests_per_level` 被自动提高并说明；受总量上限约束
    仍不足的档位会被点名；
  - 一条"整体有增益但 1→2 仍陡降"的曲线必须被描述为批处理代价，
    而"引擎安静时的吞吐下降"**绝不能**归因于服务端拒请求；
  - **容量 ≠ 并行度**：一个「引擎排队」的档位必须同时满足
    ① `peak_token_throughput` 取到它的值（容量含排队）、
    ② `engine_limited_capacity = true`、③ `parallelism_levels` 里**没有**它。
    （真机回归：把排队档位从峰值里剔除会让 460 tok/s 的机器被报成 234）
  - 饱和平台结论（`引擎已饱和` + `容量就是这个水平`）与
    `--max-num-seqs` 点名必须出现；
  - 引用的 `eff` 必须是**全档位最大值**且与档位对应
    （真机曾打印「最高 64（并发 51.06 档）」——`(档位, eff)` 元组解包反了）

---

## 安全设计

- 所有请求走 `config.py` 的**硬上限**：max context / max concurrency /
  max output tokens / max requests / max timeout，客户端无法突破。
- 每个 HTTP 请求都有 timeout + retry（指数退避）。
- 运行可取消（`DELETE /api/runs/{id}`），SSE 实时回传
  `running / warmup / testing / completed / failed / timeout`。
- API Key 只在前端输入、后端内存使用，**不落库、不进日志**，
  错误信息里也只出现 `(none)` 或脱敏形式。

---

## 本机环境注意事项（踩过的坑）

1. **`HTTP_PROXY` 会破坏 httpx 请求。**
   本机存在 `HTTP_PROXY=http://127.0.0.1:6449`，httpx 默认读环境代理，
   导致 `/v1/models`、`/metrics` 全部被代理变成 404（`/health` 偶然 200）。
   实测对照：httpx 走代理 → 404；httpx `trust_env=False` → 200；urllib → 200。
   因此 Adapter 默认 `trust_env=False`（直连）。
   若确实需要在企业代理后运行，设 `LLMB_TRUST_ENV_PROXY=1`。

2. **pip 需要绕开代理 + 用 pypi.org。**
   清华源在代理下会返回 403，表现为 `from versions: none`。
   可用做法：先清空 `HTTP_PROXY / HTTPS_PROXY / ALL_PROXY`，
   再 `pip install -i https://pypi.org/simple`。

3. **共享 venv `envs/default` 是坏的**（有 `Lib/Scripts` 但缺 `pyvenv.cfg`）。
   本项目另建了干净环境 `envs/llm-bench`。

4. **benchmark 有效性：prompt cache 会污染 Prefill 指标 —— 已验证并默认关闭。**
   真机实测 `cache_prompt` 开关造成 **6.7 倍** TTFT 差距（124ms ↔ 836ms，
   详见上文「prompt cache 实测」）。默认值已改为 `false`，且运行配置随结果展示。

5. **Vite dev server 只绑 `localhost`（IPv6 `::1`）**，用 `127.0.0.1` 连会被拒；
   需要固定 IPv4 时用 `npm run dev -- --host 127.0.0.1`。

6. **agent-browser 在本沙箱起不来 Chrome**（`exit code 3`，不写 DevToolsActivePort），
   浏览器自动化/截图不可用。替代方案：`npm run check:render`
   用 React 服务端渲染做无头校验。

7. **目标测试服务地址会变，而且会换引擎形态。**
   开发过程中先后对接过 halogen fork、原生 llama.cpp、vLLM 三类服务，
   地址也换过多次。**不要猜地址，先探测再干活** —— 同一个地址换一台机器，
   能力可能完全不同。

   四台机器的差异很关键（同一套代码必须都能测）：

   | 能力 | 原生 llama.cpp | halogen fork | **vLLM 0.29** |
   |---|---|---|---|
   | `/props` `/slots` | 有 | 无 | 无 |
   | `/tokenize` | 有 | 无 | **有（POST）** → 精确 token 构造 |
   | 原生 `/completion`（`cache_prompt` 开关） | 有 | 无 | 无 |
   | `/metrics` | 未启用（501） | `llamacpp:*` | **`vllm:*`（含 `cache_config_info`）** |
   | 每响应 `timings` 块 | 无 | 有 | 无（改用 `/metrics` 直方图交叉校验） |
   | 思考链字段 | `reasoning_content` | `reasoning_content` | **`reasoning`** |
   | 缓存开关 | `cache_prompt` | 无 | 无（prefix caching 常开） |
   | 上下文 | 262,144 | 1,048,576 | 65536 |
   | KV pool | `/health` | `halogen:*` | **`cache_config_info` 1,103,309 tokens** |

8. **路由能力探测的两个反例（踩过）**：
   - `OPTIONS` 在 llama.cpp 上对**任意路径都返回 200**（含不存在的），
     完全无法用来判断路由存在，已弃用。
   - `GET` 对 POST-only 路由在 halogen fork 上返回 `405`，但在原生
     llama.cpp 上返回 `404` —— 所以不能只靠 405 这一招。
   现在的主方法是**用无害 payload 发 POST**：
   `/tokenize` 发 `{"content":"x"}`、`/completion` 发 `{"prompt":"","n_predict":0}`
   （实测 `tokens_predicted: 0`，零成本、确定性最高）。

9. **长上下文 prefill 会把引擎打挂 —— 这是实测，不是理论风险。**
   真机上跑 128K 输入时，llama.cpp 进程直接退出（客户端收到
   `RemoteProtocolError: peer closed connection`），几秒后服务自己回来了
   （疑似 OOM 重启）。**如果测试工具把「引擎死亡」当成「输入太长」，
   它就会用一个假上限去污染报告。** 现在两者分开：

   | 现象 | 分类 | 处理 |
   |---|---|---|
   | HTTP 400 / 413 | `too_long` | 容量边界，停止加长 |
   | Connection refused | `unreachable` | 存活探测 → 中止运行，标 `partial` |

   任何一档失败都会顺手做一次 5s 超时的 `GET /v1/models` 存活检查
   （不能用默认 120s 超时 —— 一个假死的服务能把探测挂住两分钟）。

10. **固定请求超时会掐死长档位。**
    64K 输入的 prefill 需要 ~180s，默认 120s 超时在引擎还在算的时候就把请求
    掐断（实测失败样本 e2e 恰好 120.08s），随后引擎失联。
    已改为 `scaled_timeout`：按保守 tok/s 估算放宽，仍受 1800s 硬封顶。

11. **失败重试要分类型，不能一把梭。**
    「被拒绝」（400 / 连接被对端关闭）重试没有任何意义，只会浪费时间；
    只有「读超时」值得重试。早期版本对 `ConnectError` 也重试，结果在引擎已经
    死掉的情况下白等了 3 轮。

12. **一个数字要配上它的全部条件才可读。**
    「TTFT 90 秒」 alone 会让人觉得慢得离谱；配上「输入 32,851 tokens、
    cache 关闭、每档唯一标记」就知道这是 365 tok/s 的真实 prefill。
    结果页的配置条（ConfigStrip）就是为此存在的。

---

## 换引擎之后补的三件事（都是被真机逼出来的）

### 1. 唯一标记必须在**开头**，不是在末尾

引擎缓存的是**前缀**，不是整条 prompt。把唯一标记追加在末尾，前面所有内容
依然是可命中的前缀 —— 真机实测：`input=103`、**`cache_n=64`（62% 命中）**，
TTFT 543ms 里绝大部分不是冷计算。

更隐蔽的是引擎还有**对齐**（`prompt_cache.snapshot_align: 64`）：只能复用
整数个块，所以 64 这个数字正好是"对齐后的公共前缀"。把标记挪到开头之后：

| | `cache_n` | TTFT | Prefill |
|---|---|---|---|
| 标记在**末尾** | **64**（62% 命中） | 543.6 ms | 190.7 tok/s |
| 标记在**开头** | **0** | 378.0 ms | 281.9 tok/s |

这条错误是靠引擎自报的 `cache_n` 抓到的，不是靠数字"看起来不对"。

### 2. 关不掉缓存时，自动改用唯一输入

当前引擎没有原生 `/completion`，`cache_prompt` 开关不存在，而 `/health` 明确
`prompt_cache.enabled: true`。此时**重复输入必然命中缓存**，把选择权留给用户
就是在知情的情况下产出假数字。现在会在能力门控里自动打开 `unique_prompt`
并说明原因；同时把"引擎自报缓存命中"变成一条**数据警告**
（`cache_n > 0` 是引擎自己说的，不是我们猜的）。

### 3. 没有 `/tokenize` 时，用两次探针**标定**长度换算

`/tokenize` 不存在时只能按字符估算，实测偏差 **10–19%**（"8K"其实是 6.6K）。
现在先发两次不同长度的探针请求，用返回的 `prompt_tokens` 解出
`P = 开销 + 字符数 / 比例` 两个未知数，再据此构造输入：

| | 目标 512 | 目标 2048 |
|---|---|---|
| 标定前（4 字符/token） | 461（**−10%**） | 1659（**−19%**） |
| 标定后 | 516（+0.8%） | 2048（0.0%） |

真机拟合结果：**5.38 字符/token，模板开销约 62 tokens**。等级从 `estimated`
升为 `calibrated`，报告里两档都标出来。

### 附带：用引擎自报数据交叉校验我们自己的测量

fork 的每个响应都带 `timings`（`prompt_n / prompt_ms / prompt_per_second /
predicted_per_second / cache_n / disk_restore_n / draft_n / draft_n_accepted`）。
流式响应的最后两个 chunk 里也有。现在会抓下来并落进结果：

- `server` 块把 **引擎自报 prompt_ms** 与 **我们的客户端 TTFT** 并排显示
  （实测比值 0.96 —— 两边测的是同一件事，互相印证）
- `cache_hit_tokens` 取 `timings.cache_n` 与
  `usage.prompt_tokens_details.cached_tokens` 的较大值（两种服务端都可能只给一个）
- `draft_accepted_over_draft_n` 只按字面命名：这对字段的语义没有文档，
  实测既有 1.0 也有 1.1（accepted > proposed），不能当成接受率

### 4. 思考链默认就吃掉整个输出预算

该引擎 `reasoning_effort` 默认 **`xhigh`**、思考上限 32768，而
`thinking_answer_room = max(1024, 15% × max_tokens)`。实测 `max_tokens=16`
时 16 个输出 token 全是 reasoning、正文为空（`reasoning_closed_by:
"answer_room"`）。现在**探测阶段**就会提示（而不是跑完才在结果里发现），
运行前也会给一条 settings warning。

---

## 接 vLLM 节点暴露的三个问题（都是真机逼出来的）

节点：`vLLM 0.29.0.dev0 / Qwen3.8-Flash-Next / max_model_len 65536`。
接上去第一反应是「这台引擎坏了」：每次运行 `success_rate = 0`，全部样本记成失败。
实际是**我们的问题**，而且是三种不同性质的问题。

### 1. 思考链的字段名不同：`delta.reasoning` ≠ `delta.reasoning_content`

这是**阻断性** bug，症状极具误导性：

| 我们读的字段 | 引擎实际发的 | 结果 |
|---|---|---|
| `delta.reasoning_content` | `delta.reasoning` | 16 个 chunk **全部被丢弃** |

于是解析到 0 个「内容 token」→ `ok = false` → `success_rate = 0`。
**请求本来就是成功的**（`usage.completion_tokens = 16`、`finish_reason = length`），
但报告里是「全失败」。引擎没坏，是我们的解析器瞎了。

现在 `_REASONING_FIELDS = ("reasoning_content", "reasoning")` 两个都认，
流式与非流式都走同一个 helper。mock 加了 `--thinking-field` 开关，
两种拼法各有断言，这类错误不会再靠肉眼发现。

### 2. vLLM 把「有没有开 prefix caching」写在 `*_info` 指标的**标签**里

`vllm:cache_config_info{enable_prefix_caching="True", kv_cache_size_tokens="1103309", kv_cache_max_concurrency="16.83", cache_dtype="fp8", ...} 1.0`

- **`collapse()` 会把标签求和折叠掉**，这个指标在归一化后只剩一个 `1.0` ——
  信息全丢。所以能力探测改为直接读**原始 parsed samples 的 labels**。
- 这条是引擎**自述**的事实，比任何推断都硬：由此得到
  `prompt_cache_enabled = true`（无需 `/health`，vLLM 的 `/health` 是空的），
  并且**没有 API 开关能关掉它** → 触发「自动改用唯一输入」。
- 顺带拿到两个以后很有用的数：**KV pool 1,103,309 tokens**（长上下文的真天花板，
  作为只读参数展示）与 **KV 推算并发上限 16.83 路**。

### 3. 静态「没声明 ⇒ 不支持」的判断，把唯一能用的开关扔掉了

这台 vLLM 任何端点都不声明思考链控制，所以我们把 `thinking_control` 判成
`false`，于是 `enable_thinking` 被能力门控**剥掉**，而实测：

| 形式 | 结果 |
|---|---|
| `chat_template_kwargs.enable_thinking=false` | ✅ reasoning 153→**0** 字符，prompt 59→19 tokens |
| 顶层 `enable_thinking=false` | ❌ 与默认**逐字节相同**（被静默忽略） |
| `reasoning_effort=none` | ✅ 也生效 |
| `chat_template_kwargs.reasoning_effort` 非法值 | 400，并**列出**支持值：`xhigh(default)/medium/low` |

没有这个开关，`max_tokens=400` 也会被思考链吃光（实测 400 个 token 全是
reasoning、正文 0 字符），decode 测的是「思考速度」，质量测试根本拿不到答案文本。

**改法**：探测改成**行为探测**而不是读声明 —— `BaseProvider.probe_thinking_control()`
发两次 16-token 的请求（开/关各一次），比较 reasoning 字符数。
代价几百毫秒，换来的是「实测有效」而不是「未声明所以没有」。结果同时进
`/api/probe` 的 `thinking_probe`（前端有独立的「思考链实测」区块，显示
关闭前/后的 reasoning 与正文字符数）。

### 顺带修的两个

- **vLLM 指标改名**：`gpu_cache_usage_perc` → `kv_cache_usage_perc`。
  只认旧名会静默报 `None`。两个都认，并补了 `prefix_cache_hit_ratio`、
  `preemptions_total`、`prefill_kv_tokens_mean`（= prefill 实际算了多少 token，
  低于 prompt tokens 就是复用了缓存的硬证据）。
- **mock 保真度**：mock 之前忽略 `stream: false`，一律回 SSE，
  非流式路径（能力探测用）从来没被真正测过。现在两种都支持。


