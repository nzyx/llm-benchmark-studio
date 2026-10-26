import { compactInt } from '../format'
import type { Capabilities, ProbeResponse } from '../types'

/** Human labels for the engine capability flags discovery reports. */
export const CAPABILITY_LABELS: Record<keyof Capabilities, string> = {
  props: '/props 模型属性',
  slots: '/slots 槽位状态',
  tokenize: '/tokenize 精确 token 计数',
  native_completion: '/completion 原生端点',
  metrics_prometheus: 'Prometheus /metrics',
  kv_cache_metrics: 'KV cache 指标',
  request_latency_histograms: '服务端延迟直方图',
  cache_prompt_control: '可关闭 prompt cache',
  prompt_cache_enabled: '',
  speculative_decoding: '投机解码',
  vision: '视觉输入',
  tool_calls: '工具调用',
  thinking_control: '思考链控制',
}

const FAMILY_LABEL: Record<string, string> = {
  llamacpp: 'llama.cpp 系',
  vllm: 'vLLM',
  openai_compatible: '通用 OpenAI 兼容',
}

export function familyLabel(family: string): string {
  return FAMILY_LABEL[family] || family
}

interface Props {
  probe: ProbeResponse
  model: string
  modelValues: Record<string, number | null>
}

export default function ServiceDetails({ probe, model, modelValues }: Props) {
  const detection = probe.detection
  const health = probe.health || {}

  return (
    <div className="card">
      <h2>服务详情</h2>

      <dl className="kv">
        <dt>引擎家族</dt>
        <dd>
          {familyLabel(detection.family)}
          <span className="conf" title="指纹识别置信度">
            {detection.confidence}
          </span>
        </dd>
        <dt>适配器</dt>
        <dd>{probe.adapter}</dd>
        <dt>引擎版本</dt>
        <dd>{health.version?.engine || '—'}</dd>
        <dt>模型</dt>
        <dd title={model}>{model || '—'}</dd>
        <dt>上下文窗口</dt>
        <dd>{compactInt(modelValues.context_window)} tokens</dd>
        <dt>服务端输出上限</dt>
        <dd>{compactInt(modelValues.server_max_output)} tokens</dd>
        <dt>并发槽位</dt>
        <dd>{modelValues.server_slots ?? '—'}</dd>
        <dt>KV pool</dt>
        <dd>{compactInt(modelValues.server_kv_pool)} tokens</dd>
        {modelValues.server_kv_concurrency ? (
          <>
            <dt>并发上限（KV 推算）</dt>
            <dd title="KV 容量给出的上限，不等于 --max-num-seqs 的调度上限">
              {compactInt(modelValues.server_kv_concurrency)} 路
            </dd>
          </>
        ) : null}
        <dt>探测耗时</dt>
        <dd>{probe.elapsed_ms} ms</dd>
      </dl>

      <div className="cap-title">引擎能力</div>
      <div className="badges">
        {(Object.keys(CAPABILITY_LABELS) as (keyof Capabilities)[]).map(
          (key) => {
            if (key === 'prompt_cache_enabled') return null
            const on = Boolean(detection.capabilities[key])
            return (
              <span className={'badge' + (on ? '' : ' mute')} key={key}>
                {on ? '✓' : '×'} {CAPABILITY_LABELS[key]}
              </span>
            )
          }
        )}
      </div>

      {/* The thinking switch is usually not declared by any endpoint, so the
          only honest thing to show is what the two probe requests measured. */}
      {probe.thinking_probe && probe.thinking_probe.field ? (
        <div className="thinking-probe">
          <div className="cap-title">思考链实测</div>
          <div className="probe-line">
            <code>{probe.thinking_probe.field}</code>
            {probe.thinking_probe.supported === true ? (
              <span className="chip ok">生效</span>
            ) : probe.thinking_probe.supported === false ? (
              <span className="chip bad">无效</span>
            ) : (
              <span className="chip mute">未观察到思考链</span>
            )}
          </div>
          <div className="probe-detail">
            关闭前 reasoning {probe.thinking_probe.baseline_reasoning_chars ?? 0} 字符 /
            正文 {probe.thinking_probe.baseline_content_chars ?? 0} 字符 → 关闭后
            reasoning {probe.thinking_probe.off_reasoning_chars ?? 0} 字符 /
            正文 {probe.thinking_probe.off_content_chars ?? 0} 字符
          </div>
          {probe.thinks_by_default ? (
            <div className="probe-detail warn-text">
              该引擎默认先输出思考链：小 max_tokens 会被吃光、正文为空，
              建议把「启用思考链」设为关闭。
            </div>
          ) : null}
        </div>
      ) : null}

      {detection.cache_config &&
      detection.cache_config.enable_prefix_caching ? (
        <div className="probe-detail" style={{ marginTop: 10 }}>
          引擎自述缓存配置：prefix caching ={' '}
          {detection.cache_config.enable_prefix_caching}
          {detection.cache_config.cache_dtype
            ? ` · dtype ${detection.cache_config.cache_dtype}`
            : ''}
          {detection.cache_config.block_size
            ? ` · block ${detection.cache_config.block_size}`
            : ''}
        </div>
      ) : null}

      {detection.evidence.length ? (
        <>
          <div className="cap-title">判定依据</div>
          <ul className="evidence">
            {detection.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </>
      ) : null}

      {detection.notes.length ? (
        <div className="banner warn" style={{ marginTop: 12 }}>
          {detection.notes.join('；')}
        </div>
      ) : null}

      {probe.warnings.map((w, i) => (
        <div className="banner warn" key={i} style={{ marginTop: 12 }}>
          {w}
        </div>
      ))}
    </div>
  )
}
