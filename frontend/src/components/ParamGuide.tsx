import { useEffect } from 'react'
import RichText from './RichText'
import type { ParamRegistry, ParamSpec } from '../types'

const SIDE_LABEL: Record<string, string> = {
  model: '模型能力',
  input: '输入侧',
  output: '输出侧',
  sampling: '采样',
  execution: '执行控制',
}

function rangeText(p: ParamSpec): string {
  if (p.status === 'readonly') return '服务端只读'
  if (p.status === 'planned') return `${p.introduced} 接入`
  const parts: string[] = []
  if (p.min !== null && p.min !== undefined) parts.push(`≥ ${p.min}`)
  if (p.max !== null && p.max !== undefined)
    parts.push(`≤ ${p.max.toLocaleString('en-US')}`)
  const d =
    p.default === null || p.default === undefined || p.default === ''
      ? '—'
      : String(p.default)
  return `默认 ${d}${parts.length ? ` · ${parts.join(' ')}` : ''}`
}

export default function ParamGuide({
  registry,
  onClose,
}: {
  registry: ParamRegistry
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const rel = registry.relationship
  const cats = [...registry.categories].sort((a, b) => a.order - b.order)

  return (
    <div className="overlay" onClick={onClose}>
      <div className="guide" onClick={(e) => e.stopPropagation()}>
        <div className="guide-head">
          <div>
            <h2>参数速查</h2>
            <p className="sub">
              每个参数属于哪一侧、单位是什么、动了它会改变哪些指标。
            </p>
          </div>
          <button className="btn ghost" onClick={onClose}>
            关闭 (Esc)
          </button>
        </div>

        {/* ---------- the relationship ---------- */}
        <section className="guide-section">
          <h3>先把最容易混的两件事分清</h3>
          <div className="formula">{rel.formula}</div>
          <div className="glossary">
            {rel.glossary.map((g) => (
              <div className={'gcard side-' + g.side} key={g.term}>
                <div className="gcard-head">
                  <strong>{g.term}</strong>
                  <span className={'side-chip side-' + g.side}>
                    {SIDE_LABEL[g.side] ?? g.side}
                  </span>
                </div>
                <p>{g.text}</p>
              </div>
            ))}
          </div>

          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr>
                  <th>场景</th>
                  <th>输入 tokens</th>
                  <th>max_tokens</th>
                  <th>实际输出</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {rel.examples.map((ex) => (
                  <tr key={ex.label}>
                    <td>{ex.label}</td>
                    <td>{ex.input_tokens.toLocaleString('en-US')}</td>
                    <td>{ex.max_tokens.toLocaleString('en-US')}</td>
                    <td>{ex.actual_output.toLocaleString('en-US')}</td>
                    <td style={{ textAlign: 'left' }}>{ex.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ---------- full reference ---------- */}
        {cats.map((cat) => {
          const items = registry.params.filter((p) => p.category === cat.key)
          if (!items.length) return null
          return (
            <section className="guide-section" key={cat.key}>
              <h3>
                {cat.label}
                <span className="cat-count">{items.length}</span>
              </h3>
              <p className="cat-blurb">{cat.blurb}</p>
              <div className="param-list">
                {items.map((p) => (
                  <div className={'prow status-' + p.status} key={p.key}>
                    <div className="prow-head">
                      <code className="pk">{p.key}</code>
                      <strong>{p.label}</strong>
                      {p.unit ? <span className="punit">{p.unit}</span> : null}
                      <span className="prange">{rangeText(p)}</span>
                    </div>
                    <div className="prow-detail">
                      <RichText text={p.detail || p.summary} />
                    </div>
                    {p.affects.length ? (
                      <div className="prow-affects">
                        影响指标：
                        {p.affects.map((a, i) => (
                          <span className="affect" key={i}>
                            {a}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}
