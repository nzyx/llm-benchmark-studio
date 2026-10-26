/**
 * Every report must be self-describing: the same engine produces completely
 * different numbers with prompt caching on vs off (measured on the real
 * engine: TTFT 124ms vs 836ms), so the config has to travel with the result.
 */
export default function ConfigStrip({
  config,
  extra = [],
}: {
  config: Record<string, any>
  /** task-specific items, appended after the common ones */
  extra?: { k: string; v: string; warn?: boolean; hint?: string }[]
}) {
  const c = config || {}
  const opts = c.options || {}
  const items: { k: string; v: string; warn?: boolean; hint?: string }[] = [
    ...extra,
  ]

  if (c.max_tokens != null) items.push({ k: '输出上限', v: `${c.max_tokens} tok` })
  if (c.repeats != null) items.push({ k: '采样', v: `${c.repeats} 次` })
  if (c.warmup != null) items.push({ k: '预热', v: `${c.warmup} 次` })
  if (c.temperature != null) items.push({ k: '温度', v: String(c.temperature) })
  if (c.use_chat != null)
    items.push({ k: 'Chat 模板', v: c.use_chat ? '开' : '关' })
  if (c.unique_prompt) items.push({ k: '唯一输入', v: '开' })
  if (opts.seed != null) items.push({ k: 'seed', v: String(opts.seed) })
  if (c.enable_thinking != null)
    items.push({ k: '思考链', v: c.enable_thinking ? '开' : '关' })
  if (opts.cache_prompt !== undefined)
    items.push({
      k: 'prompt cache',
      v: opts.cache_prompt ? '复用' : '关闭',
      warn: Boolean(opts.cache_prompt),
      hint: opts.cache_prompt
        ? '开启了 prompt cache：TTFT / Prefill 反映的是缓存复用后的状态，不代表冷启动算力'
        : '关闭 prompt cache：每次请求都重新 prefill，这才是引擎的真实算力',
    })

  if (!items.length) return null
  return (
    <div className="config-strip">
      {items.map((it) => (
        <span
          className={'config-item' + (it.warn ? ' warn' : '')}
          key={it.k}
          title={it.hint}
        >
          <em>{it.k}</em>
          {it.v}
        </span>
      ))}
      {opts.cache_prompt ? (
        <span className="config-note">
          ⚠ 本次开启 prompt cache，TTFT / Prefill 偏乐观
        </span>
      ) : null}
    </div>
  )
}
