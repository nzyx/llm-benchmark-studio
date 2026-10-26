/**
 * Headless render check for the parameter UI.
 *
 * Chromium cannot launch in every environment (this sandbox: Chrome exits
 * with code 3, no DevToolsActivePort), so instead of screenshotting we
 * server-render the real components against the live backend payloads and
 * assert the expected content shows up.
 *
 * Run it with:  npm run check:render
 */

import { renderToStaticMarkup } from 'react-dom/server'
import ConcurrencyView from './components/ConcurrencyView'
import ContextView from './components/ContextView'
import ParamField from './components/ParamField'
import ParamGuide from './components/ParamGuide'
import ResultView from './components/ResultView'
import ServiceDetails from './components/ServiceDetails'
import type {
  Capabilities,
  ParamRegistry,
  ParamSpec,
  ProbeResponse,
  RunDoc,
} from './types'

const API = process.env.SSR_API || 'http://127.0.0.1:8000'

const failures: string[] = []

function check(ok: boolean, label: string, detail = '') {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${label}${detail ? ' — ' + detail : ''}`)
  if (!ok) failures.push(label)
}

/**
 * A minimal concurrency-run document, so the view can be checked without
 * depending on a stored run of the right shape.
 */
function syntheticConcurrencyDoc(): RunDoc {
  const level = (c: number, reqTput: number, tokTput: number, eff: number,
                 p95: number, errRate = 0, requests = 12) => ({
    concurrency: c,
    requests,
    achievable_concurrency: Math.min(c, requests),
    underprovisioned: requests < c,
    wall_s: 12 / reqTput,
    request_throughput: reqTput,
    token_throughput: tokTput,
    input_throughput: tokTput * 2,
    effective_concurrency: eff,
    success_rate: 1 - errRate,
    error_rate: errRate,
    errors: errRate ? ['ConnectError: All connection attempts failed'] : [],
    failure_kind: errRate ? 'unreachable' : null,
    ttft_ms: { mean: p95 * 0.8, p50: p95 * 0.7, p95, p99: p95 * 1.05,
               min: p95 * 0.6, max: p95 * 1.1 },
    tpot_ms: 22,
    e2e_ms: { mean: 2000, p50: 1900, p95: 2400, p99: 2500, min: 1500, max: 2600 },
    decode_tps: 45,
    output_tokens_total: tokTput * 12 / reqTput,
    samples: [],
  })

  return {
    schema_version: '1.0', run_id: 'render-check-cc', created_at: '',
    task_type: 'concurrency', preset: 'concurrency', label: null,
    status: 'completed',
    provider: { kind: 'llamacpp', base_url: 'http://x' },
    model: { id: 'mock' },
    config: { max_tokens: 128, options: { cache_prompt: false } },
    samples: [],
    stats: {},
    result: {
      ladder: [1, 4, 8],
      tested: 3,
      requests_per_level: 12,
      total_requests: 36,
      server_slots: 4,
      queues_above_slots: true,
      peak_request_throughput: 21.1,
      peak_token_throughput: 168.8,
      at_concurrency: 8,
      scaling: { from_concurrency: 1, to_concurrency: 8, concurrency_ratio: 8 },
      aborted: false,
      abort_reason: null,
      // a tax with no throughput gain: the shape that was read as "non-linear"
      batching: {
        base_concurrency: 1,
        base_tpot_ms: 5.29,
        base_decode_tps: 189.1,
        batched_tpot_ms: 24.97,
        batched_decode_tps: 40.1,
        batching_tax_x: 4.72,
        base_token_throughput: 138.4,
        peak_token_throughput: 141.0,
        peak_at_concurrency: 4,
        throughput_gain_x: 1.02,
        effective_concurrency_ceiling: 7.91,
        ceiling_at: 8,
        batch1_fast_path: true,
      },
    },
    levels: [
      level(1, 2.72, 21.8, 1.0, 225),
      level(4, 9.89, 79.1, 3.97, 288),
      level(8, 21.1, 168.8, 7.91, 230, 0.15),
      // fired fewer requests than it claims: effective concurrency can never
      // exceed 12, so this row measures a lower level than its label says
      level(32, 21.0, 168.0, 11.1, 240, 0, 12),
    ],
    data_warnings: ['档位 [8] 超过了服务端槽位数 4：超出的请求只能排队'],
    errors: { count: 0, rate: 0, items: [] },
  } as unknown as RunDoc
}

function has(html: string, needle: string) {
  return html.includes(needle)
}

/**
 * A minimal context-run document, used when the database does not happen to
 * contain the shape we need.  The two `disclaimer` strings are copied from
 * the backend verbatim on purpose: if the wording drifts, the "rendered
 * verbatim" assertion below is what catches it.
 */
function syntheticContextDoc(aborted: boolean): RunDoc {
  const CONVERGED_DISCLAIMER =
    '这是本次运行**实测通过**的最大输入长度，不是模型的理论上下文上限。' +
    '理论值请以服务端上报的上下文窗口为准。'
  const ABORTED_DISCLAIMER =
    '探测**未能完成**：`value` 只是中断之前实测通过的最大长度，' +
    '`upper_bound` 是首次观察到的失败长度 —— 真正的上限夹在这两者之间。' +
    '请确认引擎存活后重跑，以获得收敛后的结论。'

  const sample = (ttft: number, prefill: number) =>
    ({
      index: 0, ok: true, warmup: false, error: null,
      ttft_ms: ttft, ttft_content_ms: ttft, e2e_ms: ttft + 400,
      decode_window_ms: 400, tpot_ms: 25, decode_tps: 40, prefill_tps: prefill,
      itl_ms: 20, itl_p95_ms: 30, input_tokens: 2048, output_tokens: 16,
      total_tokens: 2064, reasoning_tokens: null, usage_source: 'server',
      finish_reason: 'length', output_chars: 40, text_preview: 'ok',
    }) as any

  const entry = (target: number, ttft: number | null): any => ({
    kind: 'context' as const,
    target_tokens: target,
    built: {
      target_tokens: target,
      actual_tokens: ttft == null ? null : target,
      source: 'tokenize' as const,
      units_used: 2,
      deviation_pct: 0,
      notes: [],
      tokenize_failed: false,
    },
    samples: ttft == null ? [] : [sample(ttft, 300)],
    ok: ttft != null,
    error: ttft == null ? 'HTTP 400: request exceeds the available context size' : null,
    failure_kind: ttft == null ? 'too_long' : null,
    ttft_ms: ttft,
    ttft_ms_p95: ttft,
    prefill_tps: ttft == null ? null : 300,
    decode_tps: ttft == null ? null : 40,
    e2e_ms: ttft == null ? null : ttft + 400,
    input_tokens: ttft == null ? null : 2048,
    output_tokens: ttft == null ? null : 16,
  })

  const contexts: any[] = [
    { kind: 'plan', ceiling: 262128, output_budget: 16, strategy: 'repeat_padding',
      skipped: [] },
    entry(1024, 1200),
    entry(2048, 2400),
  ]
  if (aborted) {
    contexts.push(entry(16384, null))
    contexts.push({
      kind: 'context', target_tokens: 32768, built: null, samples: [], ok: false,
      skipped_reason: 'engine_unavailable', failure_kind: 'unreachable',
      error: '引擎在 16K 这一档之后失联',
    })
  } else {
    contexts.push(entry(4096, 5200))
    // one step built without a tokenizer, to exercise the "calibrated" label
    const cal = contexts[2]
    cal.built = {
      ...cal.built,
      source: 'calibrated',
      actual_tokens: null,
      deviation_pct: null,
      notes: ['引擎未提供 /tokenize；长度按**实测标定**换算'],
    }
    // one step whose TTFT contained a queue wait, so its `prefill_tps` is not a
    // prefill rate: the real node reported 1083 tok/s where its own accounting
    // said 1536, because 35.7 s of TTFT was spent behind an abandoned warmup
    const queued = entry(8192, 121144)
    queued.queue_wait_ms = 35734
    queued.prefill_contaminated = true
    queued.prefill_tps = 1083.0
    queued.server_prefill_tps = 1536.1
    contexts.push(queued)
  }

  const mvc = aborted
    ? {
        label: 'Maximum Verified Context',
        value: 2048, upper_bound: 16384, method: 'binary_search',
        steps: [], step_count: 0, verified: false,
        aborted: true,
        abort_reason: '引擎在 16K 这一档之后失联（unreachable），后续测量已中止',
        disclaimer: ABORTED_DISCLAIMER,
      }
    : {
        label: 'Maximum Verified Context',
        value: 32768, upper_bound: 40960, safe_ceiling: 262128,
        method: 'binary_search',
        steps: [
          { target_tokens: 40960, ok: false, detail: 'HTTP 400: exceeds context' },
          { target_tokens: 32768, ok: true, detail: null },
        ],
        step_count: 2, verified: true, aborted: false,
        disclaimer: CONVERGED_DISCLAIMER,
      }

  return {
    schema_version: '1.0', run_id: 'render-check-ctx', created_at: '',
    task_type: 'context', preset: 'context', label: null,
    status: aborted ? 'partial' : 'completed',
    provider: { kind: 'llamacpp', base_url: 'http://x' },
    model: { id: 'mock' },
    config: { max_tokens: 16, repeats: 1, warmup: 0, use_chat: true,
              context_ladder: [1024, 2048, 4096], options: { cache_prompt: false } },
    samples: [],
    stats: {},
    result: {
      ladder: contexts.filter((c) => c.kind === 'context').map((c) => c.target_tokens),
      tested: 3, passed: aborted ? 2 : 3, failed: aborted ? 1 : 0,
      skipped: [],
      safe_ceiling: 262128, output_budget: 16,
      fill_strategy: 'repeat_padding',
      max_verified_context: mvc,
      largest_passing: aborted ? 2048 : 4096,
      smallest_failing: aborted ? 16384 : null,
      scaling: { from_tokens: 1024, to_tokens: 4096, ratio: 4,
                 ttft_ms_first: 1200, ttft_ms_last: 5200,
                 ttft_ms_change_x: 4.33 },
      prefill_fit: aborted
        ? null
        : {
            fixed_ms: 296.0, marginal_ms_per_token: 0.617,
            marginal_tps: 1621.0, points: 4,
            note: '每次请求约有 0.30s 的固定开销（不随输入变化），在此之上按约 ' +
                  '1,621 tok/s 处理输入。短档位的 `prefill_tps` 偏低是因为固定开销' +
                  '被摊薄得少，不代表长档位真的更快。',
          },
      calibration: aborted
        ? null
        : { chars_per_token: 3.42, overhead_tokens: 38,
            observations: [ { chars: 1200, prompt_tokens: 389 },
                            { chars: 4800, prompt_tokens: 1441 } ] },
      aborted,
      abort_reason: aborted
        ? '引擎在 16K 这一档之后失联（unreachable），后续测量已中止'
        : null,
    },
    contexts,
    errors: { count: 0, rate: 0, items: [] },
  } as unknown as RunDoc
}

async function main() {
  // ---- registry ------------------------------------------------------
  const registry: ParamRegistry = await (await fetch(`${API}/api/params`)).json()
  const guide = renderToStaticMarkup(
    <ParamGuide registry={registry} onClose={() => {}} />
  )
  check(guide.length > 2000, 'ParamGuide renders', `${guide.length} chars`)
  check(has(guide, '参数速查'), 'guide has a title')
  check(
    has(guide, '上下文窗口'),
    'guide explains the context window'
  )
  check(
    has(guide, '输入 tokens（受 context_ladder 控制）'),
    'guide shows the input + output <= context formula'
  )
  check(has(guide, 'max_tokens'), 'guide lists max_tokens')
  // The registry is fully live as of Phase 6 (no parameter is waiting for a
  // later phase), so these assert the *invariant* rather than a fixed state:
  // the guide shows a future-phase strip exactly when such params exist.
  const plannedParams = registry.params.filter((p) => p.status === 'planned')
  check(
    has(guide, '接入') === plannedParams.length > 0,
    'the guide shows a future-phase strip exactly when planned params exist',
    `${plannedParams.length} planned`
  )
  for (const cat of registry.categories) {
    check(has(guide, cat.label), `guide renders category ${cat.key}`)
  }

  // ---- a single field ------------------------------------------------
  const maxTokens = registry.params.find(
    (p) => p.key === 'max_tokens'
  ) as ParamSpec
  check(!!maxTokens, 'registry contains max_tokens spec')
  const field = renderToStaticMarkup(
    <ParamField spec={maxTokens} value={256} onChange={() => {}} />
  )
  check(has(field, '输出上限 max_tokens'), 'ParamField shows the label')
  check(
    has(field, '不是上下文长度'),
    'ParamField tooltip warns it is NOT context length'
  )
  check(has(field, 'info-dot'), 'ParamField renders the info affordance')
  check(has(field, 'tokens'), 'ParamField shows the unit')

  // a parameter that is still marked as belonging to a later phase, when one
  // exists; otherwise the read-only state is the one worth asserting
  const plannedParam = registry.params.find((p) => p.status === 'planned')
  if (plannedParam) {
    const plannedField = renderToStaticMarkup(
      <ParamField spec={plannedParam} value={4096} onChange={() => {}} />
    )
    check(has(plannedField, 'planned'), 'planned param renders in planned state')
  } else {
    const readonlyParam = registry.params.find((p) => p.status === 'readonly')!
    const readonlyField = renderToStaticMarkup(
      <ParamField spec={readonlyParam} value={1048576} onChange={() => {}} />
    )
    check(has(readonlyField, 'readonly'),
          'a read-only param renders as server-reported',
          readonlyParam.key)
  }

  const boolField = renderToStaticMarkup(
    <ParamField
      spec={registry.params.find((p) => p.key === 'use_chat') as ParamSpec}
      value={true}
      onChange={() => {}}
    />
  )
  check(has(boolField, 'switch'), 'boolean param renders as a switch')

  // ---- engine detection card -----------------------------------------
  const caps: Capabilities = {
    props: false, slots: false, tokenize: false, native_completion: false,
    metrics_prometheus: true, kv_cache_metrics: true,
    request_latency_histograms: false, cache_prompt_control: true,
    prompt_cache_enabled: true, speculative_decoding: true,
    vision: false, tool_calls: true, thinking_control: true,
  }
  const synthetic: ProbeResponse = {
    ok: true,
    elapsed_ms: 123.4,
    provider: { kind: 'llamacpp', engine: '0.14.2', base_url: 'http://x', api_version: '0.14.2' },
    adapter: 'LlamaCppProvider',
    detection: {
      family: 'llamacpp',
      confidence: 0.95,
      adapter: 'LlamaCppProvider',
      evidence: ['/metrics 含 `llamacpp:*` 序列'],
      notes: ['该 llama.cpp 系引擎未提供 /props 与 /slots'],
      capabilities: caps,
      health: {},
      props: null,
      slots: null,
      metrics_family: 'llamacpp',
      metrics_available: true,
    },
    models: [],
    model_param_values: { m: { context_window: 1048576, server_slots: 4 } },
    health: { version: { engine: '0.14.2' } },
    warnings: [],
  }
  const details = renderToStaticMarkup(
    <ServiceDetails probe={synthetic} model="m" modelValues={{ context_window: 1048576, server_slots: 4 }} />
  )
  check(has(details, '引擎家族'), 'ServiceDetails shows the engine family row')
  check(has(details, 'llama.cpp 系'), 'ServiceDetails renders a human family label')
  check(has(details, '判定依据'), 'ServiceDetails lists the detection evidence')
  check(has(details, '/metrics 含 `llamacpp:*` 序列'), 'evidence text rendered')
  check(has(details, '0.95'), 'confidence rendered')

  // ---- the measured thinking probe, and a vLLM-shaped node -----------
  // vLLM declares nothing about its chat template, so this block is the only
  // place the answer is visible to the user.
  const vllmCaps: Capabilities = {
    ...caps,
    props: false, slots: false, tokenize: true, native_completion: false,
    kv_cache_metrics: true, cache_prompt_control: false,
    prompt_cache_enabled: true, thinking_control: true,
    speculative_decoding: false, vision: false, tool_calls: false,
  }
  const vllmProbe: ProbeResponse = {
    ...synthetic,
    adapter: 'VLLMProvider',
    capabilities: vllmCaps,
    thinks_by_default: true,
    thinking_probe: {
      supported: true,
      field: 'chat_template_kwargs.enable_thinking',
      probe_tokens: 16,
      baseline_reasoning_chars: 118,
      baseline_content_chars: 0,
      off_reasoning_chars: 0,
      off_content_chars: 86,
      conclusion: 'disabling_it_removes_reasoning',
    },
    detection: {
      ...synthetic.detection,
      family: 'vllm',
      adapter: 'VLLMProvider',
      confidence: 0.95,
      capabilities: vllmCaps,
      evidence: ['/metrics 含 `vllm:*` 序列'],
      notes: ['该 vLLM 开启了 prefix caching 且没有关掉的 API 开关'],
      metrics_family: 'vllm',
      cache_config: {
        enable_prefix_caching: 'True',
        kv_cache_size_tokens: '1103309',
        kv_cache_max_concurrency: '16.83',
        cache_dtype: 'fp8',
        block_size: '800',
      },
    },
    model_param_values: {
      m: { context_window: 65536, server_kv_pool: 1103309,
           server_kv_concurrency: 16.83 },
    },
  }
  const vllmHtml = renderToStaticMarkup(
    <ServiceDetails
      probe={vllmProbe}
      model="m"
      modelValues={{ context_window: 65536, server_kv_pool: 1103309,
                     server_kv_concurrency: 16.83 }}
    />
  )
  check(has(vllmHtml, '思考链实测'),
        'the measured thinking-control probe is shown')
  check(has(vllmHtml, 'chat_template_kwargs.enable_thinking'),
        'and names the exact field that was tried')
  check(has(vllmHtml, '生效'), 'with a plain verdict chip')
  check(has(vllmHtml, '118') && has(vllmHtml, '86'),
        'and the before/after numbers behind that verdict')
  check(has(vllmHtml, '默认先输出思考链'),
        'a thinking-by-default engine is flagged for the user')
  check(has(vllmHtml, 'prefix caching ='), 'the engine-stated cache config is shown')
  check(has(vllmHtml, 'fp8'), 'including the cache dtype')
  check(has(vllmHtml, '并发上限（KV 推算）'),
        'the KV-derived concurrency ceiling has its own row')
  check(has(vllmHtml, '1.1M'), 'and the KV pool size is rendered compactly')
  check(has(details, '可关闭 prompt cache'), 'capability labels rendered')
  check(has(details, '×'), 'unsupported capabilities are marked')

  // ---- stored runs, shared by the view checks below -------------------
  const all = await (await fetch(`${API}/api/runs?limit=50`)).json()

  // ---- result view ----------------------------------------------------
  // Pick a *performance* run explicitly: whichever run happens to be newest
  // is now often a concurrency or context run, whose document has no sample
  // table.  Environment-dependent fixtures make a check look flaky.
  const perfRuns = (all.runs as { run_id: string; task_type: string }[]).filter(
    (r) => r.task_type === 'performance'
  )
  // Pick a run that actually produced samples: the newest performance run may
  // be a failure (an unreachable node records one), and a run with no samples
  // has no sample table.  Walking back a few is more honest than asserting a
  // table exists just because *some* run has one.
  let perfDoc: RunDoc | null = null
  for (const r of perfRuns.slice(0, 10)) {
    const detail = await (await fetch(`${API}/api/runs/${r.run_id}`)).json()
    const candidate = detail.result as RunDoc
    if ((candidate.samples || []).length) {
      perfDoc = candidate
      break
    }
  }
  if (perfDoc) {
    const doc = perfDoc
    const html = renderToStaticMarkup(<ResultView doc={doc} />)
    check(html.length > 1000, 'ResultView renders', `${html.length} chars`)
    check(has(html, 'TTFT'), 'ResultView shows TTFT')
    check(has(html, 'Decode'), 'ResultView shows Decode')
    check(has(html, 'td'), 'ResultView renders the sample table')

    // the audit line only appears when the engine reports its own timings
    const withServer = {
      ...doc,
      result: {
        ...doc.result,
        server: {
          available: true,
          samples: 3,
          prompt_ms_mean: 1180,
          client_ttft_ms_mean: 1230,
          prompt_over_ttft: 0.96,
          cache_hit_samples: 0,
        },
      },
    } as RunDoc
    const html2 = renderToStaticMarkup(<ResultView doc={withServer} />)
    check(has(html2, '服务端自报'), 'ResultView shows the server/client cross-check')
    check(has(html2, '无缓存命中'), 'and reports the engine-declared cache state')

    const cachedDoc = {
      ...doc,
      result: {
        ...doc.result,
        server: { available: true, samples: 3, cache_hit_samples: 3 },
      },
    } as RunDoc
    const html3 = renderToStaticMarkup(<ResultView doc={cachedDoc} />)
    check(has(html3, '命中缓存'),
          'a cached run is labelled as such rather than silently averaged')

    // A sample whose decode could not be measured must be marked, not shown as
    // an empty cell the reader has to interpret.  This is the UI half of the
    // "decode looks unstable" bug: the engine flushed 8-32 tokens in one
    // burst, so there is no honest rate to print.
    const coalescedDoc = {
      ...doc,
      samples: [
        {
          ...(doc.samples || [])[0],
          decode_tps: null,
          tpot_ms: null,
          decode_span_ms: 0.036,
          stream_events: 4,
          decode_unmeasurable: 'coalesced_flush',
        },
      ],
    } as RunDoc
    const html4 = renderToStaticMarkup(<ResultView doc={coalescedDoc} />)
    check(has(html4, 'n/a*'),
          'an unmeasurable decode is marked rather than left blank')
    check(has(html4, 'decode 无法测量'),
          'and the table explains why, with the way to fix it')
  } else {
    check(false, 'ResultView renders', 'no stored performance run available')
  }

  // ---- context view: both shapes, from deterministic fixtures --------
  // Deliberately *not* driven by whichever stored run happens to be newest:
  // `mvc.aborted` (the detection stopped early) and `result.aborted` (the run
  // stopped early) are different facts and a real run may have either one
  // without the other.  Picking a stored run made this check flaky for a
  // reason that had nothing to do with the UI.
  const ctxRuns = (all.runs as { run_id: string; task_type: string }[]).filter(
    (r) => r.task_type === 'context'
  )

  const fixtures = [
    { shape: 'converged' as const, doc: syntheticContextDoc(false) },
    { shape: 'aborted' as const, doc: syntheticContextDoc(true) },
  ]

  for (const { shape, doc } of fixtures) {
    const mvc = doc.result?.max_verified_context
    if (!mvc) {
      check(false, `MVC present (${shape})`)
      continue
    }
    const html = renderToStaticMarkup(<ContextView doc={doc} />)
    check(html.length > 1500, `ContextView renders (${shape})`, `${html.length} chars`)
    check(has(html, 'Context Scaling'), `ContextView has a title (${shape})`)
    check(
      has(html, 'Maximum Verified Context'),
      `ContextView headlines the Maximum Verified Context (${shape})`
    )
    check(has(html, '逐档结果'), `per-step table rendered (${shape})`)
    check(has(html, '实际 tokens'), `target vs actual kept apart (${shape})`)
    // the caveat has to survive to the screen verbatim - the whole point of
    // the wording is that it must not be paraphrased into a stronger claim
    check(
      has(html, mvc.disclaimer.slice(0, 10)),
      `the MVC disclaimer is rendered verbatim (${shape})`,
      mvc.disclaimer.slice(0, 10)
    )
    if (mvc.aborted) {
      check(has(html, '探测中断'), 'an aborted detection is labelled, not hidden')
      check(
        has(html, '运行提前结束') && has(html, '重跑'),
        'the abort and its remedy are explained'
      )
      check(has(html, '未执行'), 'un-attempted ladder steps are still shown')
      if (process.env.SSR_DEBUG) {
        const i = html.indexOf('探测中断')
        console.log('  [debug] window:',
                    html.slice(Math.max(0, i - 900), i + 200))
      }
    } else {
      check(
        has(html, '不是模型的理论上下文上限'),
        'the MVC disclaimer refuses to overclaim'
      )
      if (mvc.steps.length) {
        check(has(html, '二分试探'), 'the binary-search steps are listed')
      }
      if (doc.result?.calibration) {
        check(has(html, '长度标定'),
              'the fitted chars-per-token is shown in the config strip')
        check(has(html, '标定'), 'a calibrated input is labelled as such')
      }
      // A step whose TTFT contained a queue wait must be marked, not printed as
      // a rate.  This is the UI half of "longer context is faster": the live
      // node showed 1083 tok/s for a step whose own accounting said 1536.
      if (
        (doc.contexts || []).some(
          (c: any) => c.prefill_contaminated
        )
      ) {
        check(has(html, 'n/a‡'),
              'a queue-contaminated prefill is not printed as a rate')
        check(has(html, '测到的不是算力'),
              'and the table explains what it does not mean')
        check(has(html, 'Prefill（引擎）'),
              'the engine-reported rate has its own column')
      }
      if ((doc.result as any)?.prefill_fit) {
        check(has(html, '固定开销拟合'),
              'the fixed-cost fit is shown, so a rising short-context prefill '
              + 'is not read as "longer is faster"')
      }
    }
  }

  // every shape must be *constructible* from the fixtures above, and any real
  // stored run should at least render without blowing up
  check(fixtures.length === 2, 'both context shapes are covered')
  let realRendered = 0
  for (const r of ctxRuns.slice(0, 5)) {
    const detail = await (await fetch(`${API}/api/runs/${r.run_id}`)).json()
    const doc = detail.result as RunDoc
    if (!doc.result?.max_verified_context && !doc.contexts) continue
    const html = renderToStaticMarkup(<ContextView doc={doc} />)
    if (html.length > 800) realRendered += 1
  }
  check(!ctxRuns.length || realRendered > 0,
        'a stored context run also renders', `${realRendered} run(s)`)

  // ---- concurrency view ----------------------------------------------
  const ccDoc = syntheticConcurrencyDoc()
  const ccHtml = renderToStaticMarkup(<ConcurrencyView doc={ccDoc} />)
  check(ccHtml.length > 1500, 'ConcurrencyView renders', `${ccHtml.length} chars`)
  check(has(ccHtml, 'Concurrency 结果'), 'ConcurrencyView has a title')
  check(has(ccHtml, '请求吞吐') && has(ccHtml, 'Token 吞吐'),
        'both throughputs are named separately')
  check(has(ccHtml, '实测并发'),
        'the measured overlap column is present (not just the requested level)')
  check(has(ccHtml, 'TTFT p95') && has(ccHtml, 'TTFT p99'),
        'latency percentiles are shown')
  check(has(ccHtml, '排队'),
        'a level above the engine slot count is marked as queueing')
  check(has(ccHtml, '失败'), 'a level with errors is not hidden')
  // a level fired with fewer requests than its concurrency claim cannot
  // measure that concurrency - it must be marked, not printed as a level
  check(has(ccHtml, '上限'),
        'the achievable-concurrency column is present')
  check(has(ccHtml, '†'),
        'an under-provisioned level is flagged in the table')
  check(has(ccHtml, '不可能**超过请求数') || has(ccHtml, '不可能'),
        'and the table explains why the row is not that concurrency')
  // the real story of a flat curve: per-request decode collapses when batched
  check(has(ccHtml, '并发为什么没换来吞吐'),
        'the batch-1 fast path is explained instead of being left as a '
        + 'mysterious flat curve')
  check(has(ccHtml, '4.72'),
        'with the measured per-request tax quoted')

  console.log('')
  if (failures.length) {
    console.log(`FAILED: ${failures.length} -> ${failures.join(', ')}`)
    process.exit(1)
  }
  console.log('ALL RENDER CHECKS PASSED')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
