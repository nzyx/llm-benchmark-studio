import { useMemo } from 'react'
import type { ConcurrencyLevel, ConcurrencyResult, RunDoc } from '../types'
import { int, num, pct } from '../format'
import ConfigStrip from './ConfigStrip'
import ScaleChart from './ScaleChart'

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

/**
 * Two charts, because the two throughputs answer different questions:
 * requests/s shows how well the server juggles work, tokens/s is the actual
 * compute ceiling, and the latency chart is what a user feels.
 */
function LoadCharts({ levels }: { levels: ConcurrencyLevel[] }) {
  const labels = levels.map((l) => String(l.concurrency))

  const throughput = useMemo(
    () => ({
      grid: { left: 56, right: 56, top: 42, bottom: 34 },
      tooltip: { trigger: 'axis' },
      legend: { data: ['请求吞吐', 'Token 吞吐'], top: 4, textStyle: { fontSize: 12 } },
      xAxis: {
        type: 'category',
        data: labels,
        name: '并发数',
        nameLocation: 'middle',
        nameGap: 24,
        nameTextStyle: { fontSize: 11 },
        axisLabel: { fontSize: 11 },
      },
      yAxis: [
        {
          type: 'value',
          name: 'req/s',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { lineStyle: { type: 'dashed', opacity: 0.4 } },
        },
        {
          type: 'value',
          name: 'tok/s',
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 11 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: '请求吞吐',
          type: 'line',
          smooth: true,
          symbolSize: 6,
          data: levels.map((l) => l.request_throughput),
          itemStyle: { color: '#5c9a7a' },
          lineStyle: { color: '#5c9a7a', width: 2 },
        },
        {
          name: 'Token 吞吐',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbolSize: 6,
          data: levels.map((l) => l.token_throughput),
          itemStyle: { color: '#7a8fb8' },
          lineStyle: { color: '#7a8fb8', width: 2, type: 'dashed' },
        },
      ],
    }),
    [levels, labels]
  )

  const latency = useMemo(
    () => ({
      grid: { left: 56, right: 56, top: 42, bottom: 34 },
      tooltip: { trigger: 'axis' },
      legend: { data: ['TTFT p50', 'TTFT p95', 'TTFT p99'], top: 4,
                textStyle: { fontSize: 12 } },
      xAxis: {
        type: 'category',
        data: labels,
        name: '并发数',
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
      ],
      series: [
        { name: 'TTFT p50', type: 'line', smooth: true, symbolSize: 5,
          data: levels.map((l) => l.ttft_ms?.p50),
          itemStyle: { color: '#86a186' }, lineStyle: { color: '#86a186' } },
        { name: 'TTFT p95', type: 'line', smooth: true, symbolSize: 5,
          data: levels.map((l) => l.ttft_ms?.p95),
          itemStyle: { color: '#E60012' }, lineStyle: { color: '#E60012', width: 2 } },
        { name: 'TTFT p99', type: 'line', smooth: true, symbolSize: 5,
          data: levels.map((l) => l.ttft_ms?.p99),
          itemStyle: { color: '#c2a26b' },
          lineStyle: { color: '#c2a26b', type: 'dashed' } },
      ],
    }),
    [levels, labels]
  )

  return (
    <div className="chart-grid">
      <div>
        <div className="section-title">吞吐 vs 并发</div>
        <ScaleChart option={throughput} />
      </div>
      <div>
        <div className="section-title">TTFT 分位 vs 并发</div>
        <ScaleChart option={latency} />
      </div>
    </div>
  )
}

export default function ConcurrencyView({ doc }: { doc: RunDoc }) {
  const res = (doc.result || {}) as ConcurrencyResult
  const levels = (doc.levels || []) as ConcurrencyLevel[]
  const ok = levels.filter((l) => l.success_rate > 0)
  const worst = levels.reduce(
    (a, l) => (l.error_rate > (a?.error_rate ?? -1) ? l : a),
    undefined as ConcurrencyLevel | undefined
  )
  const slots = res.server_slots ?? null
  const batching = res.batching ?? null

  return (
    <div className="card">
      <h2>Concurrency 结果</h2>

      <ConfigStrip
        config={doc.config}
        extra={[
          {
            k: '并发阶梯',
            v: (res.ladder || []).join(' / '),
          },
          { k: '每档请求数', v: String(res.requests_per_level ?? '—') },
          {
            k: '服务端槽位',
            v: slots == null ? '—' : String(slots),
            warn: Boolean(res.queues_above_slots),
            hint: slots == null
              ? undefined
              : `引擎报告 ${slots} 个并发槽位；超出的请求会排队，` +
                '那些档位测到的是排队后的延迟',
          },
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
          label="峰值请求吞吐"
          value={num(res.peak_request_throughput, 2)}
          unit="req/s"
          sub={`在并发 ${res.at_concurrency ?? '—'}`}
        />
        <Metric
          label="峰值 Token 吞吐"
          value={num(res.peak_token_throughput, 1)}
          unit="tok/s"
          sub={
            res.engine_limited_capacity
              ? `实测容量：并发 ${res.at_concurrency ?? '—'} 档测得，该档引擎在排队（排队正是批处理保持满载的方式）`
              : '才反映真实算力天花板'
          }
        />
        <Metric
          label="引擎并行上限"
          value={
            res.engine_parallelism_ceiling != null
              ? num(res.engine_parallelism_ceiling, 0)
              : '—'
          }
          unit={res.engine_parallelism_ceiling != null ? '路' : ''}
          sub={
            res.engine_parallelism_ceiling != null
              ? '引擎真正同时在跑的请求数（vLLM 里就是 --max-num-seqs）'
              : '引擎未暴露 running/queued 指标'
          }
        />
        <Metric
          label="档位数"
          value={String(levels.length)}
          unit="档"
          sub={`共 ${int(res.total_requests)} 个请求`}
        />
        <Metric
          label="最高成功率"
          value={pct(Math.max(...levels.map((l) => l.success_rate), 0), 0)}
          tone={worst && worst.error_rate > 0 ? 'warn' : 'ok'}
          sub={worst && worst.error_rate > 0
            ? `最差档位失败率 ${pct(worst.error_rate, 0)}`
            : '全部成功'}
        />
        <Metric label="总耗时" value={num((doc.duration_ms ?? 0) / 1000, 1)} unit="s" />
      </div>

      {ok.length >= 2 ? <LoadCharts levels={ok} /> : null}

      <div className="section-title">逐档结果</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>并发</th>
              <th title="整档的墙钟时间，吞吐就是由它算出来的">耗时</th>
              <th
                title="实测并发度 = Σ 单请求耗时 ÷ 墙钟时间。接近 1 说明请求其实是串行的，该档吞吐不代表并发能力。注意：它把请求在引擎里排队的时间也算成了「在飞」"
              >
                实测并发
              </th>
              <th
                title="引擎自己报的同时在跑请求数（运行期间采样）。这才是真正的并行度；它明显低于左侧「实测并发」时，差额就是在引擎里排队"
              >
                引擎在跑
              </th>
              <th title="这一档能测到的并发度上限 = min(并发档位, 每档请求数)">
                上限
              </th>
              <th>请求吞吐</th>
              <th>Token 吞吐</th>
              <th>TTFT p50</th>
              <th>TTFT p95</th>
              <th>TTFT p99</th>
              <th>TPOT</th>
              <th>E2E p95</th>
              <th>成功率</th>
            </tr>
          </thead>
          <tbody>
            {levels.map((l, i) => (
              <tr
                key={i}
                className={
                  (l.error_rate > 0 ? 'bad-row' : '') +
                  (l.underprovisioned ? ' warn-row' : '') +
                  (l.engine_queued ? ' queued-row' : '') +
                  (slots != null && l.concurrency > slots ? ' over-slot' : '')
                }
                title={
                  l.underprovisioned
                    ? l.requests < l.concurrency
                      ? `只发了 ${l.requests} 个请求，最多只能有 ${l.requests} 个同时在飞 —— 这一档测不到并发 ${l.concurrency}`
                      : `实测并发度只有 ${l.effective_concurrency}（声称 ${l.concurrency}）：请求数是够的，但这一档只发了一波，头尾阶段在飞数不满把均值拉低了`
                    : slots != null && l.concurrency > slots
                      ? `超过服务端槽位 ${slots}：请求会排队`
                      : undefined
                }
              >
                <td>
                  {l.concurrency}
                  {l.underprovisioned ? <span className="warn-text"> †</span> : null}
                  {l.engine_queued ? <span className="warn-text"> ‡</span> : null}
                </td>
                <td>{num(l.wall_s, 2)} s</td>
                <td
                  className={
                    (l.effective_concurrency ?? 0) < 1.3 &&
                    l.concurrency >= 4
                      ? 'warn-text'
                      : undefined
                  }
                >
                  {num(l.effective_concurrency, 1)}
                </td>
                <td
                  className={
                    l.engine_queued ? 'warn-text' : undefined
                  }
                  title={
                    l.engine_queued
                      ? `引擎同时在跑最多 ${l.engine_running_max} 个，排队峰值 ${l.engine_waiting_max} 个 —— 左侧「实测并发」里的差额是排队，不是在跑`
                      : l.engine_running_mean != null
                        ? `引擎在运行期间自报的平均/峰值在跑请求数`
                        : '该引擎没有暴露运行中请求数指标'
                  }
                >
                  {l.engine_running_mean != null
                    ? `${num(l.engine_running_mean, 1)} / ${num(l.engine_running_max, 0)}`
                    : '—'}
                </td>
                <td
                  className={l.underprovisioned ? 'warn-text' : undefined}
                  title={
                    l.underprovisioned
                      ? `请求数 ${l.requests} < 并发档位 ${l.concurrency}`
                      : undefined
                  }
                >
                  {int(l.achievable_concurrency ?? l.concurrency)}
                </td>
                <td>{num(l.request_throughput, 2)}</td>
                <td>{num(l.token_throughput, 1)}</td>
                <td>{num(l.ttft_ms?.p50, 0)}</td>
                <td>{num(l.ttft_ms?.p95, 0)}</td>
                <td>{num(l.ttft_ms?.p99, 0)}</td>
                <td>{num(l.tpot_ms, 1)}</td>
                <td>{num(l.e2e_ms?.p95, 0)}</td>
                <td>
                  <span className={'chip ' + (l.error_rate ? 'bad' : 'ok')}>
                    {pct(l.success_rate, 0)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {levels.some((l) => l.underprovisioned) ? (
          <div className="warn-text">
            † <b>采样不足</b>，这一档没能证明自己声称的并发度，原因有两种：
            {levels.some(
              (l) => l.underprovisioned && l.requests < l.concurrency
            ) ? (
              <>
                {' '}
                <b>请求数不够</b>（发出的请求数少于并发档位，在飞数**不可能**超过请求数）；{' '}
              </>
            ) : null}
            {levels.some(
              (l) => l.underprovisioned && l.requests >= l.concurrency
            ) ? (
              <>
                {' '}
                <b>只发了一波</b>（请求数够，但每波起跑和收尾阶段在飞数不满，均值被拉低到
                ~0.7N）。
              </>
            ) : null}{' '}
            「每档请求数」保持默认值时，工具会自动按 max(12, 2×并发数) 给每档分配，
            两波可以把均值拉到 0.85–0.95N。
          </div>
        ) : null}
        {levels.some((l) => l.engine_queued) ? (
          <div className="warn-text">
            ‡ <b>引擎在排队</b>：「实测并发」把请求在引擎里排队的时间也算成了「在飞」，
            而引擎自己报的同时在跑数明显更低 ——
            这一档测到的是排队后的延迟，不是它声称的并发能力。这台引擎的同时并行上限约{' '}
            <b>{int(Math.max(...levels.map((l) => l.engine_running_max ?? 0), 0))}</b>{' '}
            个（vLLM 的 <code>--max-num-seqs</code>）。
          </div>
        ) : null}
      </div>

      {batching?.batch1_fast_path ? (
        <>
          <div className="section-title">并发为什么没换来吞吐</div>
          <div className="banner warn">
            并发 {batching.base_concurrency} 的单请求解码是{' '}
            <b>{num(batching.base_decode_tps, 1)} tok/s</b>，而 2 及以上的档位都降到
            约 <b>{num(batching.batched_decode_tps, 1)} tok/s</b>（单请求代价{' '}
            <b>{num(batching.batching_tax_x, 2)}×</b>）；总吞吐最高只有{' '}
            <b>{num(batching.peak_token_throughput, 1)} tok/s</b>（并发{' '}
            {batching.peak_at_concurrency} 档，仅基线的{' '}
            {num(batching.throughput_gain_x, 2)}×）。
            {levels.some((l) => l.engine_queued)
              ? '（注意：这张表里有档位出现了引擎排队，所以「没有排队」只对不排队的档位成立。）'
              : '引擎自报没有抢占、没有排队，'}
            所以这是它自己的批处理代价（常见于 batch-1 走快路径、投机解码只在
            batch-1 生效），不是测量误差 —— 加并发提不了吞吐。
          </div>
        </>
      ) : null}

      {levels.some((l) => l.errors?.length) ? (
        <>
          <div className="section-title">错误分析</div>
          <div className="banner error">
            {levels
              .filter((l) => l.errors?.length)
              .map((l, i) => (
                <div key={i}>
                  并发 {l.concurrency}（{pct(l.error_rate, 0)} 失败
                  {l.failure_kind ? ` / ${l.failure_kind}` : ''}）：
                  {String(l.errors[0]).slice(0, 160)}
                </div>
              ))}
          </div>
        </>
      ) : null}

      {doc.errors?.items?.length ? (
        <>
          <div className="section-title">运行错误</div>
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
