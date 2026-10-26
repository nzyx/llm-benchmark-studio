import type { RunDoc, StatBlock } from '../types'
import { int, num, pct } from '../format'
import ConfigStrip from './ConfigStrip'

function Metric({
  label,
  value,
  unit,
  sub,
}: {
  label: string
  value: string
  unit?: string
  sub?: string
}) {
  return (
    <div className="metric">
      <div className="k">{label}</div>
      <div className="v">
        {value}
        {unit ? <small>{unit}</small> : null}
      </div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  )
}

const STAT_ROWS: { key: string; label: string; digits: number }[] = [
  { key: 'ttft_ms', label: 'TTFT (ms)', digits: 1 },
  { key: 'ttft_content_ms', label: 'TTFT 首正文 (ms)', digits: 1 },
  { key: 'prefill_tps', label: 'Prefill (tok/s)', digits: 1 },
  { key: 'decode_tps', label: 'Decode (tok/s)', digits: 1 },
  { key: 'tpot_ms', label: 'TPOT (ms)', digits: 2 },
  { key: 'itl_ms', label: 'ITL (ms)', digits: 2 },
  { key: 'e2e_ms', label: 'E2E (ms)', digits: 1 },
  { key: 'output_tokens', label: 'Output tokens', digits: 0 },
]

export default function ResultView({ doc }: { doc: RunDoc }) {
  const st = doc.stats || {}
  // a context run stores one point per length; this view only ever reads
  // distribution blocks, so the union is narrowed here
  const g = (k: string): StatBlock | undefined => st[k] as StatBlock | undefined
  const primary = doc.result?.primary || {}
  const okSamples = (doc.samples || []).filter((s) => s.ok)
  const failed = (doc.samples || []).filter((s) => !s.ok)
  const preview = okSamples.length ? okSamples[okSamples.length - 1].text_preview : ''

  return (
    <div className="card">
      <h2>Performance 结果</h2>

      <ConfigStrip config={doc.config} />

      {doc.settings_warnings?.length ? (
        <div className="banner warn">
          参数已校正：{doc.settings_warnings.join('；')}
        </div>
      ) : null}

      {doc.result?.server?.available ? (
        <div className="server-check">
          <span className="server-check-title">服务端自报 vs 客户端</span>
          <span>
            prefill {num(doc.result.server.prompt_ms_mean, 0)} ms
            {doc.result.server.client_ttft_ms_mean != null
              ? ` / TTFT ${num(doc.result.server.client_ttft_ms_mean, 0)} ms`
              : ''}
          </span>
          {doc.result.server.prompt_over_ttft != null ? (
            <span className="dim">
              比值 {num(doc.result.server.prompt_over_ttft, 2)}
              （接近 1 说明两边测的是同一件事）
            </span>
          ) : null}
          <span className={'chip ' + (doc.result.server.cache_hit_samples ? 'warn' : 'ok')}>
            {doc.result.server.cache_hit_samples
              ? `${doc.result.server.cache_hit_samples}/${doc.result.server.samples} 次命中缓存`
              : '无缓存命中'}
          </span>
        </div>
      ) : null}

      {doc.data_warnings?.length ? (
        <div className="banner warn">
          {doc.data_warnings.map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      ) : null}

      <div className="metrics">
        <Metric
          label="TTFT 均值"
          value={num(primary.ttft_ms ?? g('ttft_ms')?.mean, 1)}
          unit="ms"
          sub={`p50 ${num(g('ttft_ms')?.p50, 0)} · p95 ${num(g('ttft_ms')?.p95, 0)}`}
        />
        <Metric
          label="Decode"
          value={num(primary.decode_tps ?? g('decode_tps')?.mean, 1)}
          unit="tok/s"
          sub={`p50 ${num(g('decode_tps')?.p50, 1)} · p95 ${num(g('decode_tps')?.p95, 1)}`}
        />
        <Metric
          label="Prefill"
          value={num(primary.prefill_tps ?? g('prefill_tps')?.mean, 1)}
          unit="tok/s"
          sub={`≈ input/ttft`}
        />
        <Metric
          label="TPOT"
          value={num(primary.tpot_ms ?? g('tpot_ms')?.mean, 2)}
          unit="ms"
          sub={`ITL ${num(g('itl_ms')?.mean, 2)} ms`}
        />
        <Metric
          label="E2E 延迟"
          value={num(primary.e2e_ms ?? g('e2e_ms')?.mean, 0)}
          unit="ms"
          sub={`p95 ${num(g('e2e_ms')?.p95, 0)}`}
        />
        <Metric
          label="Output tokens"
          value={num(primary.output_tokens ?? g('output_tokens')?.mean, 0)}
          sub={`input ${num(g('input_tokens')?.mean, 0)}`}
        />
        <Metric
          label="成功率"
          value={pct(doc.result?.success_rate, 0)}
          sub={`${doc.result?.samples_ok ?? 0}/${doc.result?.samples_total ?? 0}`}
        />
        <Metric label="总耗时" value={num((doc.duration_ms ?? 0) / 1000, 1)} unit="s" />
      </div>

      <div className="section-title">统计分布</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>指标</th>
              <th>count</th>
              <th>mean</th>
              <th>median</th>
              <th>p50</th>
              <th>p95</th>
              <th>p99</th>
              <th>min</th>
              <th>max</th>
            </tr>
          </thead>
          <tbody>
            {STAT_ROWS.map((r) => {
              const s = g(r.key)
              if (!s) return null
              return (
                <tr key={r.key}>
                  <td>{r.label}</td>
                  <td>{int(s.count, '0')}</td>
                  <td>{num(s.mean, r.digits)}</td>
                  <td>{num(s.median, r.digits)}</td>
                  <td>{num(s.p50, r.digits)}</td>
                  <td>{num(s.p95, r.digits)}</td>
                  <td>{num(s.p99, r.digits)}</td>
                  <td>{num(s.min, r.digits)}</td>
                  <td>{num(s.max, r.digits)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="section-title">逐次请求（{doc.samples?.length ?? 0} 次）</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>TTFT</th>
              <th>首正文</th>
              <th>Decode</th>
              <th>TPOT</th>
              <th>E2E</th>
              <th>in/out</th>
              <th>usage</th>
            </tr>
          </thead>
          <tbody>
            {(doc.samples || []).map((s, i) => (
              <tr key={i} className={s.ok ? '' : 'bad-row'}>
                <td>{s.warmup ? 'warm' : s.index + 1}</td>
                <td>{num(s.ttft_ms, 0)}</td>
                <td>{num(s.ttft_content_ms, 0)}</td>
                <td
                  title={
                    s.decode_unmeasurable === 'single_flush'
                      ? '整个输出被合并成一次 flush 送达，首末 token 的时间差为 0，无法据此计算解码速度'
                      : s.decode_unmeasurable
                        ? '多个 token 事件挤在同一次读取里到达，时间无法分辨，无法据此计算解码速度'
                        : undefined
                  }
                >
                  {s.decode_tps == null && s.decode_unmeasurable
                    ? 'n/a*'
                    : num(s.decode_tps, 1)}
                </td>
                <td>{num(s.tpot_ms, 1)}</td>
                <td>{num(s.e2e_ms, 0)}</td>
                <td>
                  {int(s.input_tokens)}/{int(s.output_tokens)}
                </td>
                <td>{s.ok ? s.usage_source : (s.error || 'failed').slice(0, 28)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(doc.samples || []).some((s) => s.decode_unmeasurable) ? (
          <div className="warn-text">
            * 该样本的 decode 无法测量：引擎把输出合并送达，token
            到达时间不可分辨。已置空而非报一个假值；把 max_tokens
            调大（≥64）后重测通常可恢复。
          </div>
        ) : null}
      </div>

      {preview ? (
        <>
          <div className="section-title">最后一次输出预览</div>
          <div className="preview">{preview}</div>
        </>
      ) : null}

      {failed.length ? (
        <>
          <div className="section-title">错误分析</div>
          <div className="banner error">
            {failed.length} 次请求失败：
            <br />
            {failed
              .slice(0, 5)
              .map((s, i) => (
                <span key={i}>
                  · {String(s.error).slice(0, 160)}
                  <br />
                </span>
              ))}
          </div>
        </>
      ) : null}
    </div>
  )
}
