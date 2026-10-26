import { useMemo } from 'react'
import type {
  ContextEntry,
  ContextResult,
  MaxVerifiedContext,
  RunDoc,
} from '../types'
import { humanTokens, int, num } from '../format'
import ConfigStrip from './ConfigStrip'
import ScaleChart from './ScaleChart'

const FAILURE_LABEL: Record<string, string> = {
  too_long: '超出容量',
  unreachable: '引擎失联',
  timeout: '超时',
  auth: '认证失败',
  server_error: '服务端错误',
  unknown: '未知错误',
}

const SKIP_LABEL: Record<string, string> = {
  engine_unavailable: '引擎失联，未执行',
  beyond_first_failure: '已知更长的档位会失败，未执行',
}

function Metric({
  label,
  value,
  unit,
  sub,
  tone,
}: {
  label: string
  value: string
  unit?: string
  sub?: string
  tone?: 'ok' | 'bad' | 'warn'
}) {
  return (
    <div className={'metric' + (tone ? ' ' + tone : '')}>
      <div className="k">{label}</div>
      <div className="v">
        {value}
        {unit ? <small>{unit}</small> : null}
      </div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* charts                                                              */
/* ------------------------------------------------------------------ */
function ScalingCharts({ good }: { good: ContextEntry[] }) {
  const option = useMemo(() => {
    const labels = good.map((c) => humanTokens(c.target_tokens))
    return {
      grid: { left: 56, right: 56, top: 42, bottom: 34 },
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['TTFT', 'Prefill'],
        top: 4,
        textStyle: { fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: labels,
        name: '输入 tokens',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { fontSize: 11 },
        axisLabel: { fontSize: 11 },
      },
      yAxis: [
        {
          type: 'value',
          name: 'TTFT (ms)',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { lineStyle: { type: 'dashed', opacity: 0.4 } },
        },
        {
          type: 'value',
          name: 'Prefill (tok/s)',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: 'TTFT',
          type: 'line',
          smooth: true,
          symbolSize: 6,
          data: good.map((c) => c.ttft_ms),
          itemStyle: { color: '#E60012' },
          lineStyle: { color: '#E60012', width: 2 },
        },
        {
          name: 'Prefill',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbolSize: 6,
          data: good.map((c) => c.prefill_tps),
          itemStyle: { color: '#7a8fb8' },
          lineStyle: { color: '#7a8fb8', width: 2, type: 'dashed' },
        },
      ],
    }
  }, [good])

  const decode = useMemo(() => {
    const labels = good.map((c) => humanTokens(c.target_tokens))
    return {
      grid: { left: 56, right: 24, top: 34, bottom: 34 },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Decode', 'E2E'], top: 4, textStyle: { fontSize: 12 } },
      xAxis: {
        type: 'category',
        data: labels,
        name: '输入 tokens',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { fontSize: 11 },
        axisLabel: { fontSize: 11 },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Decode (tok/s)',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { lineStyle: { type: 'dashed', opacity: 0.4 } },
        },
        {
          type: 'value',
          name: 'E2E (ms)',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: 'Decode',
          type: 'line',
          smooth: true,
          symbolSize: 6,
          data: good.map((c) => c.decode_tps),
          itemStyle: { color: '#5c9a7a' },
          lineStyle: { color: '#5c9a7a', width: 2 },
        },
        {
          name: 'E2E',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbolSize: 6,
          data: good.map((c) => c.e2e_ms),
          itemStyle: { color: '#c2a26b' },
          lineStyle: { color: '#c2a26b', width: 2, type: 'dashed' },
        },
      ],
    }
  }, [good])

  return (
    <div className="chart-grid">
      <div>
        <div className="section-title">TTFT / Prefill vs 输入长度</div>
        <ScaleChart option={option} />
      </div>
      <div>
        <div className="section-title">Decode / E2E vs 输入长度</div>
        <ScaleChart option={decode} />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Maximum Verified Context                                            */
/* ------------------------------------------------------------------ */
function VerifiedContext({ mvc }: { mvc: MaxVerifiedContext }) {
  const aborted = Boolean(mvc.aborted)
  return (
    <div className={'mvc-card' + (aborted ? ' aborted' : '')}>
      <div className="mvc-head">
        <span className="mvc-title">Maximum Verified Context</span>
        <span className={'chip ' + (aborted ? 'warn' : mvc.verified ? 'ok' : 'mute')}>
          {aborted ? '探测中断' : mvc.verified ? '已收敛' : '未收敛'}
        </span>
      </div>

      <div className="mvc-value">
        {mvc.value == null ? '—' : humanTokens(mvc.value)}
        <span className="unit">tokens</span>
        {mvc.upper_bound ? (
          <span className="mvc-band">
            （上限 ≤ {humanTokens(mvc.upper_bound)}，真实值夹在这之间）
          </span>
        ) : null}
      </div>

      <div className="mvc-meta">
        方法 {mvc.method} · 步数 {mvc.step_count}
        {mvc.safe_ceiling ? ` · 安全上限 ${humanTokens(mvc.safe_ceiling)}` : ''}
        {mvc.bracket_from_ladder ? ' · 复用了阶梯夹逼' : ''}
      </div>

      {aborted && mvc.abort_reason ? (
        <div className="banner warn" style={{ marginTop: 10 }}>
          {mvc.abort_reason}
        </div>
      ) : null}

      {mvc.steps.length ? (
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table>
            <thead>
              <tr>
                <th>二分试探</th>
                <th>结果</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {mvc.steps.map((s, i) => (
                <tr key={i} className={s.ok ? '' : 'bad-row'}>
                  <td>{humanTokens(s.target_tokens)}</td>
                  <td>{s.ok ? '通过' : '失败'}</td>
                  <td className="dim">{String(s.detail ?? '').slice(0, 120) || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="mvc-disclaimer">ⓘ {mvc.disclaimer}</div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* main view                                                           */
/* ------------------------------------------------------------------ */
export default function ContextView({ doc }: { doc: RunDoc }) {
  const res = (doc.result || {}) as ContextResult
  const all = (doc.contexts || []) as ContextEntry[]
  const entries = all.filter((c) => c.kind === 'context')
  const tested = entries.filter((c) => c.samples?.length)
  const good = tested
    .filter((c) => c.ok)
    .sort((a, b) => a.target_tokens - b.target_tokens)
  const skipped = entries.filter((c) => c.skipped_reason)
  const factor = res.scaling?.ttft_ms_change_x ?? null
  const fit = res.prefill_fit ?? null
  const anyContaminated = entries.some((c) => c.prefill_contaminated)

  return (
    <div className="card">
      <h2>Context Scaling 结果</h2>

      <ConfigStrip
        config={doc.config}
        extra={[
          { k: '阶梯', v: (res.ladder || []).map((t) => humanTokens(t)).join(' / ') },
          {
            k: '构造方式',
            v: res.fill_strategy === 'corpus' ? '多样语料' : '重复填充',
          },
          {
            k: '输出预留',
            v: `${res.output_budget ?? 0} tok`,
            hint: '输入长度与输出长度共用同一个上下文窗口，可用输入上限已为输出预留空间',
          },
          ...(res.calibration
            ? [
                {
                  k: '长度标定',
                  v: `${num(res.calibration.chars_per_token, 2)} 字符/token`,
                  hint:
                    `引擎没有 /tokenize，长度换算由 ${res.calibration.observations.length} 次` +
                    `探针实测得出；模板开销约 ${num(res.calibration.overhead_tokens, 0)} tokens。` +
                    '实际 tokens 以服务端 usage 为准。',
                },
              ]
            : []),
        ]}
      />

      {doc.settings_warnings?.length ? (
        <div className="banner warn">
          参数已校正：{doc.settings_warnings.join('；')}
        </div>
      ) : null}

      {res.aborted && res.abort_reason ? (
        <div className="banner warn">
          <b>运行提前结束（部分完成）</b>
          <div>{res.abort_reason}</div>
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
          label="最大通过档位"
          value={humanTokens(res.largest_passing)}
          unit="tok"
          tone="ok"
          sub="实测确实跑通的最大输入"
        />
        <Metric
          label="首次失败档位"
          value={humanTokens(res.smallest_failing)}
          unit="tok"
          tone={res.smallest_failing ? 'bad' : undefined}
          sub="第一次被拒绝或中断的长度"
        />
        <Metric
          label="TTFT 增长"
          value={factor == null ? '—' : `${num(factor, 2)}×`}
          sub={
            res.scaling?.from_tokens
              ? `${humanTokens(res.scaling.from_tokens)} → ${humanTokens(
                  res.scaling.to_tokens
                )}（输入 ${num(res.scaling.ratio, 1)}×）`
              : undefined
          }
        />
        <Metric
          label="点位"
          value={`${good.length}`}
          unit="个"
          sub={`共测 ${tested.length} 档，跳过 ${skipped.length} 档`}
        />
        <Metric label="总耗时" value={num((doc.duration_ms ?? 0) / 1000, 1)} unit="s" />
      </div>

      {res.max_verified_context ? (
        <VerifiedContext mvc={res.max_verified_context} />
      ) : null}

      {good.length >= 2 ? <ScalingCharts good={good} /> : null}

      <div className="section-title">逐档结果</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>目标输入</th>
              <th>实际 tokens</th>
              <th>来源</th>
              <th>TTFT</th>
              <th title="引擎自报的 prefill 时间，用来交叉校验客户端的 TTFT">
                服务端 prefill
              </th>
              <th title="input_tokens / ttft —— 含固定开销，短档位天然偏低">
                Prefill
              </th>
              <th title="input_tokens / 引擎自报 prompt_ms —— 不含客户端开销与排队">
                Prefill（引擎）
              </th>
              <th>Decode</th>
              <th>E2E</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((c, i) => {
              const b = c.built
              const skippedReason = c.skipped_reason
              return (
                <tr key={i} className={c.ok ? '' : 'bad-row'}>
                  <td>{humanTokens(c.target_tokens)}</td>
                  <td>{b?.actual_tokens == null ? '—' : int(b.actual_tokens)}</td>
                  <td>
                    {b?.source === 'tokenize' ? (
                      '精确'
                    ) : b?.source === 'calibrated' ? (
                      <span
                        className="warn-text"
                        title="引擎没有 /tokenize，长度换算由两次探针实测标定"
                      >
                        标定
                      </span>
                    ) : b?.source === 'estimated' ? (
                      <span className="warn-text" title="引擎未提供 /tokenize，按字符估算">
                        估算
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>{num(c.ttft_ms, 0)}</td>
                  <td title="引擎自报 prompt_ms">{num(c.server_prompt_ms, 0)}</td>
                  <td
                    title={
                      c.prefill_contaminated
                        ? `这一档的 TTFT 里有 ${num(
                            (c.queue_wait_ms ?? 0) / 1000,
                            1
                          )}s 不是 prefill（排队 / 重试 / 缓存恢复）—— ` +
                          '它测的是等待而不是算力，请参考右侧引擎自报的 prefill 速度'
                        : undefined
                    }
                  >
                    {c.prefill_contaminated ? (
                      <span className="warn-text">n/a‡</span>
                    ) : (
                      num(c.prefill_tps, 1)
                    )}
                  </td>
                  <td>{num(c.server_prefill_tps, 1)}</td>
                  <td>{num(c.decode_tps, 1)}</td>
                  <td>{num(c.e2e_ms, 0)}</td>
                  <td>
                    {skippedReason ? (
                      <span className="chip mute" title={SKIP_LABEL[skippedReason]}>
                        未执行
                      </span>
                    ) : c.ok ? (
                      <span className="chip ok">通过</span>
                    ) : (
                      <span
                        className="chip bad"
                        title={String(c.error ?? '').slice(0, 200)}
                      >
                        {FAILURE_LABEL[c.failure_kind || 'unknown'] || '失败'}
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {anyContaminated ? (
          <div className="warn-text">
            ‡ 该档的 TTFT 里混入了非 prefill 的等待（上一档请求被中断后引擎仍在算、
            后续请求排队，或重试 / 缓存恢复），所以 <code>n/a</code> 表示
            「测到的不是算力」。同一行的「Prefill（引擎）」是引擎自报的
            prefill 时间算出来的，那一列才可比。
          </div>
        ) : null}
        {fit ? (
          <div className="dim" style={{ marginTop: 8 }}>
            固定开销拟合：<b>{num((fit.fixed_ms ?? 0) / 1000, 2)}s</b> / 请求 +
            <b> {num(fit.marginal_tps, 0)} tok/s</b> 边际速率（{fit.points} 个干净档位）。
            {fit.note}
          </div>
        ) : null}
      </div>

      {skipped.length ? (
        <>
          <div className="section-title">未执行的档位</div>
          <ul className="evidence">
            {skipped.map((c, i) => (
              <li key={i}>
                {humanTokens(c.target_tokens)} tokens —— {SKIP_LABEL[c.skipped_reason!] ||
                  c.skipped_reason}
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {doc.errors?.items?.length ? (
        <>
          <div className="section-title">错误分析</div>
          <div className="banner error">
            {doc.errors.items.slice(0, 5).map((e, i) => (
              <div key={i}>· {String(e.message).slice(0, 200)}</div>
            ))}
          </div>
        </>
      ) : null}
    </div>
  )
}
