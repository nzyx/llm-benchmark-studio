export interface ModelInfo {
  id: string
  max_model_len?: number | null
  context_length?: number | null
  owned_by?: string | null
  max_tokens_cap?: number | null
  max_tokens_default?: number | null
  meta?: Record<string, unknown>
}

export interface ProviderInfo {
  kind: string
  engine?: string | null
  base_url: string
  api_version?: string | null
}

export type EngineFamily = 'llamacpp' | 'vllm' | 'openai_compatible'

export interface Capabilities {
  props: boolean
  slots: boolean
  tokenize: boolean
  native_completion: boolean
  metrics_prometheus: boolean
  kv_cache_metrics: boolean
  request_latency_histograms: boolean
  cache_prompt_control: boolean
  prompt_cache_enabled: boolean | null
  speculative_decoding: boolean
  vision: boolean
  tool_calls: boolean
  thinking_control: boolean
}

export interface Detection {
  family: EngineFamily
  confidence: number
  adapter: string
  evidence: string[]
  notes: string[]
  capabilities: Capabilities
  health: Record<string, any>
  props: Record<string, any> | null
  slots: Record<string, any>[] | null
  metrics_family: string | null
  metrics_available: boolean
  /** the engine's own default thinking strength, when it reports one */
  reasoning_effort_default?: string | null
  /** engine-stated cache setup (vLLM publishes it as an `*_info` metric) */
  cache_config?: Record<string, string>
}

/** Result of the behavioural `enable_thinking` probe. */
export interface ThinkingProbe {
  supported: boolean | null
  field?: string
  probe_tokens?: number
  baseline_reasoning_chars?: number
  baseline_content_chars?: number
  off_reasoning_chars?: number
  off_content_chars?: number
  conclusion?: string
  error?: string
}

export interface ProbeResponse {
  ok: boolean
  elapsed_ms: number
  provider: ProviderInfo
  adapter: string
  detection: Detection
  /** measured capabilities, which supersede the statically inferred ones */
  capabilities?: Capabilities
  thinking_probe?: ThinkingProbe | null
  thinks_by_default?: boolean | null
  models: ModelInfo[]
  model_param_values: Record<string, Record<string, number | null>>
  health: Record<string, any>
  warnings: string[]
}

/** How the input for a context step was built. */
export interface BuiltInput {
  target_tokens: number
  /** exact token count when a tokenizer was available, else null */
  actual_tokens: number | null
  source: 'tokenize' | 'calibrated' | 'estimated'
  units_used: number
  deviation_pct: number | null
  notes: string[]
  /** the engine claims /tokenize but the call failed - a transport symptom */
  tokenize_failed: boolean
}

/**
 * The engine's text->token ratio, measured with two probe requests. Only
 * needed when there is no `/tokenize` endpoint, so the fitted overhead is
 * worth keeping: it is what turns "4 characters per token, roughly" into a
 * number measured on *this* engine.
 */
export interface Calibration {
  chars_per_token: number
  overhead_tokens: number
  observations: { chars: number; prompt_tokens: number }[]
}

export interface ContextEntry {
  kind: 'context'
  target_tokens: number
  built: BuiltInput | null
  samples: Sample[]
  ok: boolean
  error: string | null
  errors?: string[]
  /** too_long | unreachable | timeout | auth | server_error | unknown */
  failure_kind?: string | null
  ttft_ms: number | null
  ttft_ms_p95: number | null
  prefill_tps: number | null
  /** largest non-prefill wait inside TTFT at this step (queue / retry / restore) */
  queue_wait_ms?: number | null
  /** true when this step's `prefill_tps` cannot be read as a prefill rate */
  prefill_contaminated?: boolean | null
  decode_tps: number | null
  e2e_ms: number | null
  input_tokens: number | null
  output_tokens: number | null
  /* server-reported, for cross-checking our own client-side numbers */
  server_prompt_ms?: number | null
  server_prefill_tps?: number | null
  cache_hit_tokens?: number | null
  /** present when the step was never attempted */
  skipped_reason?: 'engine_unavailable' | 'beyond_first_failure'
  /** scored samples after warmups were excluded (context steps only) */
  scored_samples?: number
  /** how many samples were dropped from the statistics as warmups */
  warmup_excluded?: number
}

export interface MaxVerifiedContext {
  label: string
  /**
   * Largest input length this run *observed to work*. Null when nothing was
   * verified. Never a claim about the model's theoretical limit.
   */
  value: number | null
  /** smallest length observed to fail, when one was seen */
  upper_bound?: number | null
  lower_bound?: number
  safe_ceiling?: number
  method: string
  steps: { target_tokens: number; ok: boolean; detail: string | null }[]
  step_count: number
  verified: boolean
  aborted?: boolean
  abort_reason?: string | null
  bracket_from_ladder?: boolean
  disclaimer: string
}

export interface ScalingSummary {
  from_tokens?: number
  to_tokens?: number
  ratio?: number
  ttft_ms_first?: number
  ttft_ms_last?: number
  ttft_ms_change_x?: number
  prefill_tps_first?: number
  prefill_tps_last?: number
  prefill_tps_change_x?: number
  e2e_ms_first?: number
  e2e_ms_last?: number
  e2e_ms_change_x?: number
}

/**
 * Splits prefill cost into a fixed part and a marginal rate.  Without it a
 * `prefill_tps` column necessarily *rises* while a fixed per-request cost is
 * amortised, which reads as "longer context is faster".
 */
export interface PrefillFit {
  fixed_ms: number
  marginal_ms_per_token: number
  marginal_tps: number
  points: number
  note: string
}

export interface ContextResult {
  ladder: number[]
  tested: number
  passed: number
  failed: number
  skipped: number[]
  safe_ceiling: number | null
  output_budget: number | null
  fill_strategy: string | null
  max_verified_context: MaxVerifiedContext | null
  largest_passing: number | null
  smallest_failing: number | null
  scaling: ScalingSummary
  prefill_fit?: PrefillFit | null
  calibration: Calibration | null
  aborted: boolean
  abort_reason: string | null
}

export interface Percentiles {
  mean: number | null
  p50: number | null
  p95: number | null
  p99: number | null
  min: number | null
  max: number | null
}

/** One concurrency level: a batch of simultaneous requests, described as a batch. */
export interface ConcurrencyLevel {
  concurrency: number
  requests: number
  /**
   * The most requests this level could ever have in flight.  A "concurrency
   * 128" level fired with 12 requests can never exceed 12, so without this the
   * table looks like the engine topped out when it was the harness that ran
   * short.
   */
  achievable_concurrency?: number | null
  underprovisioned?: boolean
  /** wall clock of the whole level - the only honest basis for throughput */
  wall_s: number
  request_throughput: number
  token_throughput: number
  input_throughput: number
  /**
   * Little's law on the samples: Σ per-request duration ÷ wall clock. If a
   * "concurrency 8" level reports ~1, the requests were really serialised.
   *
   * It counts a request as in flight for its whole life, **including** time
   * queued inside the engine - so it is an upper bound on parallelism, not a
   * measurement of it.  Compare against the `engine_*` fields below.
   */
  effective_concurrency: number | null
  /* ---- sampled from the engine's own gauges *while the level ran* ----
   * A client cannot tell "32 ran at once" from "16 ran and 16 queued": both
   * look like 32 in flight.  Only the engine can, and only if it is asked
   * during the load (a before/after snapshot sees an idle engine).
   */
  engine_running_max?: number | null
  engine_running_mean?: number | null
  engine_waiting_max?: number | null
  /** the engine queued requests at this level: it is not a parallelism reading */
  engine_queued?: boolean
  /** share of `effective_concurrency` that was engine queueing */
  queue_share?: number | null
  /**
   * Share of a request's lifetime spent before its first token.  Token
   * throughput is end-to-end, so this wait is in its denominator - and it
   * overlaps away as concurrency rises, which is why the curve can climb
   * without decode getting any faster.  At high concurrency it is mostly
   * engine queueing (see `engine_waiting_max`), not prefill.
   */
  ttft_share?: number | null
  success_rate: number
  error_rate: number
  errors: (string | null)[]
  failure_kind?: string | null
  ttft_ms: Percentiles
  tpot_ms: number | null
  e2e_ms: Percentiles
  decode_tps: number | null
  output_tokens_total: number
  samples: Sample[]
}

/** Where per-request decode collapses under batching, versus concurrency 1. */
export interface BatchingProfile {
  base_concurrency: number
  base_tpot_ms: number
  base_decode_tps: number | null
  batched_tpot_ms: number
  batched_decode_tps: number | null
  batching_tax_x: number
  base_token_throughput: number
  peak_token_throughput: number | null
  peak_at_concurrency: number | null
  throughput_gain_x: number | null
  effective_concurrency_ceiling: number
  ceiling_at: number | null
  batch1_fast_path: boolean
}

export interface ConcurrencyResult {
  ladder: number[]
  tested: number
  requests_per_level: number
  total_requests: number
  server_slots: number | null
  queues_above_slots: boolean
  peak_request_throughput: number | null
  /**
   * Highest aggregate token throughput at the offered load that produced it -
   * the box's measured capacity.  Levels where the engine queued are included:
   * queueing is how the batch stays full, so it is how the capacity is reached.
   * Read `engine_limited_capacity` to know whether that was the case.
   */
  peak_token_throughput: number | null
  at_concurrency: number | null
  /** the capacity figure came from a level the engine had to queue */
  engine_limited_capacity?: boolean
  /** levels that can support a "this is N-way parallel" claim */
  parallelism_levels?: number[]
  /** best throughput among levels that demonstrated their own concurrency */
  parallel_peak_token_throughput?: number | null
  scaling: Record<string, any>
  batching?: BatchingProfile | null
  /**
   * The engine's own parallelism ceiling, taken from its running-request gauge.
   * The client's `effective_concurrency` cannot express this: queueing inflates
   * it exactly in the regime where the ceiling matters.  `null` = not exposed.
   */
  engine_parallelism_ceiling?: number | null
  engine_queued_levels?: number[]
  /**
   * Our counted output tokens ÷ the engine's own generation count.  Well below
   * 1 means our token accounting is losing tokens (an engine whose stream
   * carries no `usage`), and the token throughput is understated.
   */
  token_coverage?: number | null
  aborted: boolean
  abort_reason: string | null
}

/* ---------------- parameter registry ---------------- */

export type ParamCategoryKey =
  | 'model'
  | 'input'
  | 'output'
  | 'sampling'
  | 'execution'

export interface ParamSpec {
  key: string
  label: string
  category: ParamCategoryKey
  kind: 'int' | 'float' | 'bool' | 'text' | 'select' | 'readonly'
  default: unknown
  unit: string | null
  min: number | null
  max: number | null
  step: number | null
  choices: string[] | null
  summary: string
  detail: string
  affects: string[]
  introduced: string
  status: 'active' | 'planned' | 'readonly'
}

export interface ParamCategory {
  key: ParamCategoryKey
  label: string
  blurb: string
  order: number
}

export interface ParamRelationship {
  headline: string
  formula: string
  glossary: { term: string; side: string; text: string }[]
  examples: {
    label: string
    input_tokens: number
    max_tokens: number
    actual_output: number
    note: string
  }[]
}

export interface ParamRegistry {
  categories: ParamCategory[]
  params: ParamSpec[]
  relationship: ParamRelationship
  limits: Record<string, number>
}

/* ---------------- results ---------------- */

export interface StatBlock {
  count: number
  mean: number | null
  median: number | null
  p50: number | null
  p95: number | null
  p99: number | null
  min: number | null
  max: number | null
  std: number | null
}

export interface Sample {
  index: number
  ok: boolean
  warmup: boolean
  error: string | null
  /** stable classification of `error`, see backend app/errors.py */
  error_kind?: string | null
  /** the engine started answering but the stream broke before it finished */
  truncated?: boolean
  /**
   * Identity of the content this sample sent. Two samples sharing one
   * fingerprint cannot be independent on an engine with a prefix cache, so
   * this is what makes "are these numbers comparable?" checkable.
   */
  input_hash?: string
  input_chars?: number
  /* ---- server-reported, when the engine exposes a timings block ---- */
  server_timings?: boolean
  server_prompt_ms?: number | null
  server_prefill_tps?: number | null
  server_decode_tps?: number | null
  server_predicted_ms?: number | null
  /**
   * Prompt tokens the engine served from its own prompt cache. Non-zero means
   * this sample's prefill/TTFT is not cold compute - a fact reported by the
   * engine, not an inference from a suspicious number.
   */
  cache_hit_tokens?: number | null
  draft_tokens?: number | null
  draft_accepted?: number | null
  ttft_ms: number | null
  ttft_content_ms: number | null
  e2e_ms: number | null
  decode_window_ms: number | null
  /** how many stream events actually carried a token; `tpot_ms`/`decode_tps`
   *  are derived from *when these arrived*, not from `output_tokens`. */
  stream_events?: number | null
  /** first-token-to-last-token window, the basis of `tpot_ms` */
  decode_span_ms?: number | null
  /** why decode could not be measured: the tokens were not separately timed */
  decode_unmeasurable?: 'single_flush' | 'coalesced_flush' | null
  /** the slice of TTFT that was NOT prefill: queueing / retry / cache restore.
   *  Only set when the engine reports its own prefill time. */
  queue_wait_ms?: number | null
  /** true when `queue_wait_ms` is large enough that `prefill_tps` describes
   *  waiting rather than computing */
  prefill_contaminated?: boolean | null
  tpot_ms: number | null
  decode_tps: number | null
  prefill_tps: number | null
  itl_ms: number | null
  itl_p95_ms: number | null
  input_tokens: number | null
  output_tokens: number | null
  total_tokens: number | null
  reasoning_tokens: number | null
  reasoning_chars?: number
  content_chars?: number
  usage_source: string
  finish_reason: string | null
  output_chars: number
  text_preview: string
  unique_prompt?: boolean
}

export interface RunDoc {
  schema_version: string
  run_id: string
  created_at: string
  task_type: string
  preset: string | null
  label: string | null
  status: string
  provider: ProviderInfo
  model: ModelInfo
  model_param_values?: Record<string, number | null>
  config: Record<string, any>
  settings_warnings?: string[]
  /** data-quality guards raised by the task, e.g. inconsistent input lengths */
  data_warnings?: string[]
  started_at: string | null
  finished_at: string | null
  duration_ms: number | null
  samples: Sample[]
  stats: Record<string, StatBlock | SeriesBlock>
  result: {
    samples_total?: number
    samples_ok?: number
    success_rate?: number
    primary?: Record<string, number | null>
    /* context-task fields, mirrored to the top level for convenience */
    ladder?: number[]
    passed?: number
    failed?: number
    largest_passing?: number | null
    smallest_failing?: number | null
    safe_ceiling?: number | null
    max_verified_context?: MaxVerifiedContext | null
    scaling?: ScalingSummary
    calibration?: Calibration | null
    aborted?: boolean
    abort_reason?: string | null
    /* concurrency-task fields, mirrored to the top level */
    requests_per_level?: number
    total_requests?: number
    server_slots?: number | null
    queues_above_slots?: boolean
    peak_request_throughput?: number | null
    peak_token_throughput?: number | null
    at_concurrency?: number | null
    /** server-reported timings vs our client-side measurement, when available */
    server?: {
      available: boolean
      samples?: number
      prompt_ms_mean?: number
      client_ttft_ms_mean?: number
      prompt_over_ttft?: number
      cache_hit_samples?: number
      /**
       * `draft_n_accepted / draft_n` as reported by the engine. The semantics
       * of the pair are undocumented (observed at both 1.0 and 1.1), so treat
       * it as a rough indicator only.
       */
      draft_accepted_over_draft_n?: number
    }
  }
  /** per-context points, present on context runs */
  contexts?: (ContextEntry | { kind: 'plan'; [k: string]: any })[]
  /** per-concurrency-level batches, present on concurrency runs */
  levels?: ConcurrencyLevel[]
  /** one entry per task when a run carries several ("full" preset) */
  task_results?: Record<string, RunDoc>
  errors: { count: number; rate: number; items: { at?: string; message: string }[] }
  base_url?: string
}

/** A scaling series: one point per context length, not a distribution. */
export interface SeriesBlock {
  count: number
  points: { x: number; y: number }[]
  min: number | null
  max: number | null
  mean: number | null
}

export interface RunSummary {
  run_id: string
  created_at: string
  task_type: string
  preset: string | null
  label: string | null
  engine: string | null
  model_id: string | null
  status: string
  schema_version: string
  base_url: string | null
}

export interface ProgressEvent {
  status: string
  phase?: string
  done?: number
  total?: number
  message?: string
  extra?: any
  final?: boolean
  run_id?: string
}
