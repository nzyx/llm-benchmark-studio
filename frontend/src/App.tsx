import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ResultView from './components/ResultView'
import ConcurrencyView from './components/ConcurrencyView'
import ContextView from './components/ContextView'
import ParamField from './components/ParamField'
import ParamGuide from './components/ParamGuide'
import {
  apiHealth,
  createRun,
  deleteRun,
  getParams,
  getRun,
  listPresets,
  listRuns,
  openRunEvents,
  probe,
} from './api'
import { int, statusLabel, statusTone, timeAgo } from './format'
import ServiceDetails from './components/ServiceDetails'
import type {
  EngineFamily,
  ParamRegistry,
  ProbeResponse,
  ProgressEvent,
  RunDoc,
  RunSummary,
} from './types'

const DEFAULT_BASE = 'http://127.0.0.1:8080'
const STORAGE_BASE = 'llmb.base_url'

/**
 * Params the UI edits, split by where they go in the request body.
 *
 * A run carries one payload per task, so a key that only one task understands
 * must not be sent to the other: the backend would report it as an unknown
 * parameter and the warning would be noise, not information.
 */
const COMMON_TASK_KEYS = [
  'prompt',
  'use_chat',
  'max_tokens',
  'temperature',
  'warmup',
  'repeats',
  'cache_prompt',
  'unique_prompt',
  'seed',
  'enable_thinking',
] as const
const PERF_ONLY_KEYS = ['stream'] as const
const CONTEXT_ONLY_KEYS = [
  'context_ladder',
  'context_fill',
  'detect_max_context',
] as const
const CONCURRENCY_ONLY_KEYS = [
  'concurrency_ladder',
  'requests_per_level',
] as const
const TASK_KEYS = [
  ...COMMON_TASK_KEYS,
  ...PERF_ONLY_KEYS,
  ...CONTEXT_ONLY_KEYS,
  ...CONCURRENCY_ONLY_KEYS,
] as const
const RUN_KEYS = ['timeout_s', 'retries'] as const

const TASK_LABEL: Record<string, string> = {
  performance: 'Performance',
  context: 'Context Scaling',
  concurrency: 'Concurrency',
}

/** Build a task payload: only the keys that task actually understands. */
function pick(
  values: Record<string, any>,
  keys: readonly string[]
): Record<string, any> {
  const out: Record<string, any> = {}
  for (const k of keys) {
    const v = values[k]
    if (v === '' || v === undefined || v === null) continue
    out[k] = v
  }
  return out
}

export default function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [registry, setRegistry] = useState<ParamRegistry | null>(null)
  const [guideOpen, setGuideOpen] = useState(false)

  const [baseUrl, setBaseUrl] = useState(
    () => localStorage.getItem(STORAGE_BASE) || DEFAULT_BASE
  )
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)

  const [probing, setProbing] = useState(false)
  const [probeRes, setProbeRes] = useState<ProbeResponse | null>(null)
  const [probeErr, setProbeErr] = useState<string | null>(null)
  const [model, setModel] = useState('')
  // lets the user override detection when the fingerprint guessed wrong
  const [family, setFamily] = useState<'auto' | EngineFamily>('auto')

  const [presets, setPresets] = useState<Record<string, any>>({})
  const [preset, setPreset] = useState('quick')
  // Only what the user actually edited.  The form used to be seeded with the
  // registry defaults and then post *all* of them, which silently overrode the
  // preset's tuned values - measured: the Concurrency preset asks for
  // max_tokens 128 but every run went out with the registry default 256, so
  // each request cost twice what the preset intended.
  const [touched, setTouched] = useState<Record<string, any>>({})

  const [running, setRunning] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [runDoc, setRunDoc] = useState<RunDoc | null>(null)
  // which task's section of the document is on screen ("full" runs carry two)
  const [activeTask, setActiveTask] = useState<string | null>(null)
  const [runErr, setRunErr] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const [history, setHistory] = useState<RunSummary[]>([])
  const esRef = useRef<EventSource | null>(null)

  const refreshHistory = useCallback(async () => {
    try {
      setHistory((await listRuns()).runs)
    } catch {
      /* backend may still be starting */
    }
  }, [])

  useEffect(() => {
    apiHealth()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))
    listPresets()
      .then((r) => setPresets(r.presets || {}))
      .catch(() => {})
    getParams()
      .then((reg) => setRegistry(reg))
      .catch(() => {})
    refreshHistory()
    return () => esRef.current?.close()
  }, [refreshHistory])

  // What each control shows and what the run sends: registry defaults, then the
  // selected preset's tuned values, then whatever the user typed.  Making the
  // preset an actual layer (instead of posting the defaults over it) keeps the
  // form honest - the box shows the value that will really be used.
  const values = useMemo(() => {
    const out: Record<string, any> = {}
    for (const p of registry?.params ?? []) {
      if (p.status !== 'active') continue
      if ([...TASK_KEYS, ...RUN_KEYS].includes(p.key as any)) out[p.key] = p.default
    }
    const cfg = presets[preset] || {}
    for (const t of (cfg.tasks as string[]) || []) {
      Object.assign(out, cfg[t] || {})
    }
    return { ...out, ...touched }
  }, [registry, presets, preset, touched])

  const specOf = useCallback(
    (key: string) => registry?.params.find((p) => p.key === key) ?? null,
    [registry]
  )

  const planned = useMemo(
    () => (registry?.params ?? []).filter((p) => p.status === 'planned'),
    [registry]
  )

  const setValue = (key: string, v: unknown) =>
    setTouched((prev) => ({ ...prev, [key]: v }))

  const loadDoc = (doc: RunDoc | null) => {
    setRunDoc(doc)
    setActiveTask(doc ? doc.task_type : null)
  }

  const doProbe = async () => {
    setProbing(true)
    setProbeErr(null)
    setProbeRes(null)
    loadDoc(null)
    localStorage.setItem(STORAGE_BASE, baseUrl)
    try {
      const res = await probe(baseUrl, apiKey)
      setProbeRes(res)
      if (res.models.length) setModel(res.models[0].id)
    } catch (e: any) {
      setProbeErr(e.message || String(e))
    } finally {
      setProbing(false)
    }
  }

  const startRun = async () => {
    setRunErr(null)
    loadDoc(null)
    setProgress(null)
    setNote(null)
    setRunning(true)
    try {
      const base = pick(values, COMMON_TASK_KEYS)
      const performance = { ...base, ...pick(values, PERF_ONLY_KEYS) }
      const context = { ...base, ...pick(values, CONTEXT_ONLY_KEYS) }
      const concurrency = { ...base, ...pick(values, CONCURRENCY_ONLY_KEYS) }
      const res = await createRun({
        base_url: baseUrl,
        api_key: apiKey || null,
        model: model || null,
        preset,
        family: family === 'auto' ? null : family,
        timeout_s: values.timeout_s ?? null,
        retries: values.retries ?? null,
        performance,
        context,
        concurrency,
      })
      setRunId(res.run_id)
      setNote(res.note || null)

      const es = openRunEvents(res.run_id)
      esRef.current = es
      es.onmessage = async (m) => {
        let ev: ProgressEvent
        try {
          ev = JSON.parse(m.data)
        } catch {
          return
        }
        setProgress(ev)
        if (ev.final) {
          es.close()
          esRef.current = null
          setRunning(false)
          try {
            loadDoc((await getRun(res.run_id)).result)
          } catch (e: any) {
            setRunErr(e.message || String(e))
          }
          refreshHistory()
        }
      }
      es.onerror = () => {
        /* server closes the stream when the run ends */
      }
    } catch (e: any) {
      setRunning(false)
      setRunErr(e.message || String(e))
    }
  }

  const cancelRun = async () => {
    if (!runId) return
    esRef.current?.close()
    esRef.current = null
    try {
      await deleteRun(runId)
    } catch {
      /* ignore */
    }
    setRunning(false)
    setProgress({ status: 'cancelled', message: '已取消' })
  }

  const loadHistoryRun = async (id: string) => {
    setRunErr(null)
    try {
      const r = await getRun(id)
      loadDoc(r.result)
      setRunId(id)
      setProgress({ status: r.result.status, message: '历史记录' })
    } catch (e: any) {
      setRunErr(e.message || String(e))
    }
  }

  const health = probeRes?.health || {}
  const modelValues = probeRes?.model_param_values?.[model] || {}
  const modelInfo = probeRes?.models.find((m) => m.id === model)
  const total = progress?.total || 0
  const done = progress?.done || 0
  const pctDone = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0

  // a "full" run carries several task sections; the switcher picks one
  const presetTasks: string[] = presets[preset]?.tasks || []
  const runsContext = presetTasks.includes('context')
  const runsConcurrency = presetTasks.includes('concurrency')
  const taskResults = (runDoc?.task_results || {}) as Record<string, RunDoc>
  const taskNames = Object.keys(taskResults)
  const shownDoc = (activeTask && taskResults[activeTask]) || runDoc
  const shownTask = shownDoc?.task_type
  const shownIsContext = Boolean(shownDoc && (shownTask === 'context' || shownDoc.contexts))
  const shownIsConcurrency = Boolean(
    shownDoc && (shownTask === 'concurrency' || shownDoc.levels)
  )

  const activeSpec = (key: string) => {
    const s = specOf(key)
    return s && s.status === 'active' ? s : null
  }

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Local LLM Benchmark Studio</h1>
          <p className="sub">
            本地部署大模型 HTTP 服务的可视化 Benchmark 工作台 · Phase 6
          </p>
        </div>
        <div className="header-actions">
          <button
            className="btn ghost bordered"
            onClick={() => setGuideOpen(true)}
            disabled={!registry}
          >
            参数速查
          </button>
          <span className="backend-chip">
            <span
              className={
                'dot ' + (backendOk === null ? '' : backendOk ? 'ok' : 'bad')
              }
            />
            {backendOk === null
              ? '连接后端…'
              : backendOk
                ? '后端已连接'
                : '后端未连接'}
          </span>
        </div>
      </div>

      {probeErr ? <div className="banner error">探测失败：{probeErr}</div> : null}
      {runErr ? <div className="banner error">运行失败：{runErr}</div> : null}
      {note ? <div className="banner warn">{note}</div> : null}

      <div className="grid">
        {/* ---------------- left column ---------------- */}
        <div className="col">
          <div className="card">
            <h2>1 · 模型连接与服务探测</h2>
            <div className="field">
              <label>API 地址（OpenAI Compatible）</label>
              <input
                className="mono"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://127.0.0.1:8080"
                spellCheck={false}
              />
            </div>
            <div className="field">
              <label>API Key（可选，仅后端内存使用，不落库）</label>
              <div className="row">
                <input
                  className="mono"
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-... 或留空"
                  spellCheck={false}
                />
                <button
                  className="btn"
                  style={{ width: 74, flex: 'none' }}
                  onClick={() => setShowKey((v) => !v)}
                  type="button"
                >
                  {showKey ? '隐藏' : '显示'}
                </button>
              </div>
            </div>
            <button className="btn primary" onClick={doProbe} disabled={probing}>
              {probing ? <span className="spinner" /> : null}
              {probing ? '探测中…' : '探测服务'}
            </button>
            <div className="hint">
              读取 <span className="mono">/v1/models</span> 与{' '}
              <span className="mono">/health</span>，不发推理请求。
            </div>
          </div>

          {probeRes ? (
            <ServiceDetails
              probe={probeRes}
              model={model}
              modelValues={modelValues}
            />
          ) : null}

          <div className="card">
            <h2>
              2 · 运行 Benchmark
              <button
                className="btn ghost"
                onClick={() => setGuideOpen(true)}
                disabled={!registry}
                style={{ float: 'right', marginTop: -4 }}
              >
                参数速查
              </button>
            </h2>

            <div className="field">
              <label>Preset</label>
              <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                {Object.keys(presets).length ? (
                  Object.entries(presets).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v.label} — {v.description}
                    </option>
                  ))
                ) : (
                  <option value="quick">Quick Test</option>
                )}
              </select>
            </div>

            <div className="field">
              <label>引擎适配器</label>
              <select
                value={family}
                onChange={(e) =>
                  setFamily(e.target.value as 'auto' | EngineFamily)
                }
              >
                <option value="auto">
                  自动识别
                  {probeRes
                    ? `（当前：${probeRes.detection.family}）`
                    : ''}
                </option>
                <option value="llamacpp">强制 llama.cpp 系</option>
                <option value="vllm">强制 vLLM</option>
                <option value="openai_compatible">强制通用 OpenAI 兼容</option>
              </select>
              <div className="pfield-sub">
                识别错了可以手动指定；强制指定会跳过指纹检测。
              </div>
            </div>

            <div className="field">
              <label>模型</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {probeRes?.models.length ? (
                  probeRes.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))
                ) : (
                  <option value="">（请先探测服务）</option>
                )}
              </select>
            </div>

            <div className="pgroup">
              <div className="pgroup-head">
                <span className="side-chip side-input">输入侧</span>
                <span className="pgroup-blurb">
                  决定喂给模型读多少 → 影响 Prefill / TTFT
                </span>
              </div>
              {activeSpec('use_chat') ? (
                <ParamField
                  spec={activeSpec('use_chat')!}
                  value={values.use_chat}
                  onChange={(v) => setValue('use_chat', v)}
                  disabled={running}
                />
              ) : null}
              {activeSpec('prompt') ? (
                <ParamField
                  spec={activeSpec('prompt')!}
                  value={values.prompt ?? ''}
                  onChange={(v) => setValue('prompt', v)}
                  disabled={running}
                />
              ) : null}
              {activeSpec('unique_prompt') ? (
                <ParamField
                  spec={activeSpec('unique_prompt')!}
                  value={values.unique_prompt}
                  onChange={(v) => setValue('unique_prompt', v)}
                  disabled={running}
                />
              ) : null}
            </div>

            <div className="pgroup">
              <div className="pgroup-head">
                <span className="side-chip side-output">输出侧</span>
                <span className="pgroup-blurb">
                  决定让模型写多少 → 影响 Decode / E2E
                </span>
              </div>
              {activeSpec('max_tokens') ? (
                <ParamField
                  spec={activeSpec('max_tokens')!}
                  value={values.max_tokens}
                  onChange={(v) => setValue('max_tokens', v)}
                  disabled={running}
                />
              ) : null}
            </div>

            <div className="pgroup">
              <div className="pgroup-head">
                <span className="side-chip side-sampling">采样</span>
                <span className="pgroup-blurb">决定模型写出什么内容</span>
              </div>
              {activeSpec('temperature') ? (
                <ParamField
                  spec={activeSpec('temperature')!}
                  value={values.temperature}
                  onChange={(v) => setValue('temperature', v)}
                  disabled={running}
                />
              ) : null}
              {activeSpec('seed') ? (
                <ParamField
                  spec={activeSpec('seed')!}
                  value={values.seed ?? ''}
                  onChange={(v) => setValue('seed', v)}
                  disabled={running}
                />
              ) : null}
              {activeSpec('enable_thinking') ? (
                <ParamField
                  spec={activeSpec('enable_thinking')!}
                  value={values.enable_thinking}
                  onChange={(v) => setValue('enable_thinking', v)}
                  disabled={running}
                />
              ) : null}
            </div>

            {runsContext ? (
              <div className="pgroup">
                <div className="pgroup-head">
                  <span className="side-chip side-input">上下文测试</span>
                  <span className="pgroup-blurb">
                    逐档加长**输入**，测 Prefill / TTFT 随长度的变化
                  </span>
                </div>
                {activeSpec('context_ladder') ? (
                  <ParamField
                    spec={activeSpec('context_ladder')!}
                    value={values.context_ladder ?? ''}
                    onChange={(v) => setValue('context_ladder', v)}
                    disabled={running}
                  />
                ) : null}
                {activeSpec('context_fill') ? (
                  <ParamField
                    spec={activeSpec('context_fill')!}
                    value={values.context_fill}
                    onChange={(v) => setValue('context_fill', v)}
                    disabled={running}
                  />
                ) : null}
                {activeSpec('detect_max_context') ? (
                  <ParamField
                    spec={activeSpec('detect_max_context')!}
                    value={values.detect_max_context}
                    onChange={(v) => setValue('detect_max_context', v)}
                    disabled={running}
                  />
                ) : null}
              </div>
            ) : null}

            {runsConcurrency ? (
              <div className="pgroup">
                <div className="pgroup-head">
                  <span className="side-chip side-execution">并发测试</span>
                  <span className="pgroup-blurb">
                    逐档同时打 N 个请求，测吞吐与延迟劣化
                  </span>
                </div>
                {activeSpec('concurrency_ladder') ? (
                  <ParamField
                    spec={activeSpec('concurrency_ladder')!}
                    value={values.concurrency_ladder ?? ''}
                    onChange={(v) => setValue('concurrency_ladder', v)}
                    disabled={running}
                  />
                ) : null}
                {activeSpec('requests_per_level') ? (
                  <ParamField
                    spec={activeSpec('requests_per_level')!}
                    value={values.requests_per_level}
                    onChange={(v) => setValue('requests_per_level', v)}
                    disabled={running}
                  />
                ) : null}
              </div>
            ) : null}

            <div className="pgroup">
              <div className="pgroup-head">
                <span className="side-chip side-execution">执行控制</span>
                <span className="pgroup-blurb">决定怎么跑、跑几遍</span>
              </div>
              <div className="row">
                {activeSpec('warmup') ? (
                  <ParamField
                    spec={activeSpec('warmup')!}
                    value={values.warmup}
                    onChange={(v) => setValue('warmup', v)}
                    disabled={running}
                  />
                ) : null}
                {activeSpec('repeats') ? (
                  <ParamField
                    spec={activeSpec('repeats')!}
                    value={values.repeats}
                    onChange={(v) => setValue('repeats', v)}
                    disabled={running}
                  />
                ) : null}
              </div>
              <div className="row">
                {activeSpec('timeout_s') ? (
                  <ParamField
                    spec={activeSpec('timeout_s')!}
                    value={values.timeout_s}
                    onChange={(v) => setValue('timeout_s', v)}
                    disabled={running}
                  />
                ) : null}
                {activeSpec('retries') ? (
                  <ParamField
                    spec={activeSpec('retries')!}
                    value={values.retries}
                    onChange={(v) => setValue('retries', v)}
                    disabled={running}
                  />
                ) : null}
              </div>
              {activeSpec('stream') ? (
                <ParamField
                  spec={activeSpec('stream')!}
                  value={values.stream}
                  onChange={(v) => setValue('stream', v)}
                  disabled={running}
                />
              ) : null}
              {activeSpec('cache_prompt') ? (
                <ParamField
                  spec={activeSpec('cache_prompt')!}
                  value={values.cache_prompt}
                  onChange={(v) => setValue('cache_prompt', v)}
                  disabled={running}
                />
              ) : null}
            </div>

            {planned.length ? (
              <div className="planned-strip">
                <span className="planned-title">后续阶段参数</span>
                {planned.map((p) => (
                  <span className="tag planned" key={p.key} title={p.summary}>
                    {p.label} · {p.introduced}
                  </span>
                ))}
              </div>
            ) : null}

            {running ? (
              <button className="btn" onClick={cancelRun}>
                取消运行
              </button>
            ) : (
              <button
                className="btn primary"
                onClick={startRun}
                disabled={!probeRes || !model}
              >
                Run Full Benchmark
              </button>
            )}
            <div className="hint">
              已实现 Performance / Context / Concurrency 三个任务；质量测试在
              Phase 7 接入。跑之前建议点右上角「参数速查」确认口径。
            </div>
          </div>
        </div>

        {/* ---------------- right column ---------------- */}
        <div className="col">
          {progress ? (
            <div className="card">
              <div className="progress-head">
                <span className={'chip ' + statusTone(progress.status)}>
                  {['running', 'testing', 'warmup', 'connect'].includes(
                    progress.status
                  ) ? (
                    <span className="spinner" />
                  ) : null}
                  {statusLabel(progress.status)}
                </span>
                <span className="hint" style={{ margin: 0 }}>
                  {progress.phase ? `${progress.phase} · ` : ''}
                  {total > 0 ? `${done}/${total}` : ''}
                </span>
              </div>
              <div className="bar">
                <span style={{ width: `${pctDone}%` }} />
              </div>
              <div className="hint">
                {progress.message || '—'}
                {progress.extra?.last?.ttft_ms != null
                  ? ` · 本次 TTFT ${int(progress.extra.last.ttft_ms)}ms / Prefill ${
                      progress.extra.last.prefill_tps ?? '—'
                    } tok/s`
                  : ''}
              </div>
            </div>
          ) : null}

          {shownDoc ? (
            <>
              {taskNames.length > 1 ? (
                <div className="task-tabs">
                  {taskNames.map((t) => (
                    <button
                      key={t}
                      className={'tab' + (t === activeTask ? ' on' : '')}
                      onClick={() => setActiveTask(t)}
                    >
                      {TASK_LABEL[t] || t}
                      <span className={'chip ' + statusTone(taskResults[t].status)}>
                        {statusLabel(taskResults[t].status)}
                      </span>
                    </button>
                  ))}
                </div>
              ) : null}
              {shownIsContext ? (
                <ContextView doc={shownDoc} />
              ) : shownIsConcurrency ? (
                <ConcurrencyView doc={shownDoc} />
              ) : (
                <ResultView doc={shownDoc} />
              )}
            </>
          ) : null}

          {!runDoc && !progress ? (
            <div className="card">
              <h2>结果</h2>
              <div className="empty">
                还没有运行记录。先在左侧「探测服务」，然后点「Run Full Benchmark」。
                不懂参数就点右上角「参数速查」。
              </div>
            </div>
          ) : null}

          <div className="card">
            <h2>3 · Benchmark 历史</h2>
            {history.length ? (
              <div className="hlist">
                {history.map((r) => (
                  <div
                    key={r.run_id}
                    className={'hitem' + (r.run_id === runId ? ' active' : '')}
                    onClick={() => loadHistoryRun(r.run_id)}
                  >
                    <div className="hl">
                      <div className="t">{r.model_id || r.run_id}</div>
                      <div className="m">
                        {r.preset || r.task_type} · {timeAgo(r.created_at)} ·{' '}
                        {r.run_id}
                      </div>
                    </div>
                    <span className={'chip ' + statusTone(r.status)}>
                      {statusLabel(r.status)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty">暂无历史记录。</div>
            )}
          </div>
        </div>
      </div>

      {guideOpen && registry ? (
        <ParamGuide registry={registry} onClose={() => setGuideOpen(false)} />
      ) : null}
    </div>
  )
}
