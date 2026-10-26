import RichText from './RichText'
import type { ParamSpec } from '../types'

export function InfoDot({ spec }: { spec: ParamSpec }) {
  return (
    <span className="tip-wrap" tabIndex={0}>
      <span className="info-dot" aria-label={`${spec.label} 说明`}>
        i
      </span>
      <span className="tip" role="tooltip">
        <span className="tip-title">
          {spec.label}
          {spec.unit ? <em> · {spec.unit}</em> : null}
        </span>
        <span className="tip-body">
          <RichText text={spec.detail || spec.summary} />
        </span>
        {spec.affects.length ? (
          <span className="tip-affects">
            影响指标：
            {spec.affects.map((a, i) => (
              <span className="affect" key={i}>
                {a}
              </span>
            ))}
          </span>
        ) : null}
      </span>
    </span>
  )
}

interface Props {
  spec: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
  /** value reported by the server, for kind === 'readonly' */
  readOnlyValue?: number | null
  disabled?: boolean
}

export default function ParamField({
  spec,
  value,
  onChange,
  readOnlyValue,
  disabled,
}: Props) {
  const planned = spec.status === 'planned'

  if (spec.kind === 'readonly') {
    return (
      <div className="pfield readonly">
        <div className="pfield-head">
          <label>{spec.label}</label>
          <InfoDot spec={spec} />
        </div>
        <div className="pfield-value">
          {readOnlyValue === null || readOnlyValue === undefined
            ? '探测后显示'
            : readOnlyValue.toLocaleString('en-US')}
          {spec.unit ? <em>{spec.unit}</em> : null}
        </div>
      </div>
    )
  }

  return (
    <div className={'pfield' + (planned ? ' planned' : '')}>
      <div className="pfield-head">
        <label>{spec.label}</label>
        <InfoDot spec={spec} />
        {spec.unit ? <span className="pfield-unit">{spec.unit}</span> : null}
        {planned ? <span className="tag planned">{spec.introduced} 接入</span> : null}
      </div>

      {spec.kind === 'bool' && spec.default === null ? (
        // tri-state: unset means "whatever the server defaults to", which is
        // not the same as false - worth keeping explicit
        <div className="tristate">
          {[
            { label: '服务端默认', v: null },
            { label: '开启', v: true },
            { label: '关闭', v: false },
          ].map((opt) => (
            <button
              type="button"
              key={opt.label}
              className={
                'tri-opt' +
                ((value ?? null) === opt.v ? ' on' : '')
              }
              disabled={disabled || planned}
              onClick={() => onChange(opt.v)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      ) : spec.kind === 'bool' ? (
        <button
          type="button"
          className={'switch' + (value ? ' on' : '')}
          disabled={disabled || planned}
          onClick={() => onChange(!value)}
          aria-pressed={!!value}
        >
          <span className="knob" />
          <span className="switch-label">{value ? '开启' : '关闭'}</span>
        </button>
      ) : spec.kind === 'select' ? (
        <select
          value={String(value ?? spec.default ?? '')}
          disabled={disabled || planned}
          onChange={(e) => onChange(e.target.value)}
        >
          {(spec.choices || []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : spec.kind === 'text' ? (
        <textarea
          rows={3}
          value={String(value ?? '')}
          disabled={disabled || planned}
          placeholder={spec.summary}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <input
          type="number"
          className="mono"
          value={value === null || value === undefined ? '' : String(value)}
          min={spec.min ?? undefined}
          max={spec.max ?? undefined}
          step={spec.step ?? undefined}
          disabled={disabled || planned}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === '') return onChange('')
            onChange(spec.kind === 'int' ? parseInt(raw, 10) : parseFloat(raw))
          }}
        />
      )}

      <div className="pfield-sub">{spec.summary}</div>
    </div>
  )
}
