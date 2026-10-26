export function num(v: number | null | undefined, digits = 1, dash = '—'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash
  if (!Number.isFinite(v)) return dash
  return v.toFixed(digits)
}

export function int(v: number | null | undefined, dash = '—'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash
  return Math.round(v).toLocaleString('en-US')
}

export function compactInt(v: number | null | undefined, dash = '—'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash
  const n = Math.round(v)
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `${(n / 1000).toFixed(0)}K`
  return n.toLocaleString('en-US')
}

export function pct(v: number | null | undefined, digits = 1, dash = '—'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash
  return `${(v * 100).toFixed(digits)}%`
}

/** Token counts in the units people actually say out loud: 1K, 128K, 1M. */
export function humanTokens(v: number | null | undefined, dash = '—'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash
  const n = Math.round(v)
  if (n >= 1_048_576 && n % (1024 * 1024) === 0) return `${n / (1024 * 1024)}M`
  if (n >= 1_048_576) return `${(n / (1024 * 1024)).toFixed(1)}M`
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`
  if (n >= 1024) return `${(n / 1024).toFixed(1)}K`
  return String(n)
}

export function timeAgo(iso: string): string {
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const diff = Date.now() - t
  const m = Math.floor(diff / 60000)
  if (m < 1) return '刚刚'
  if (m < 60) return `${m} 分钟前`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h} 小时前`
  const d = Math.floor(h / 24)
  if (d < 30) return `${d} 天前`
  return new Date(iso).toLocaleDateString('zh-CN')
}

export const STATUS_LABEL: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  warmup: '预热中',
  testing: '测试中',
  connect: '连接中',
  completed: '已完成',
  // measured something real, but stopped before finishing what was asked
  partial: '部分完成',
  failed: '失败',
  timeout: '超时',
  cancelled: '已取消',
  pending: '待运行',
}

export function statusLabel(status: string | undefined): string {
  if (!status) return '—'
  return STATUS_LABEL[status] ?? status
}

export function statusTone(status: string | undefined): string {
  switch (status) {
    case 'completed':
      return 'ok'
    case 'partial':
      return 'warn'
    case 'failed':
    case 'timeout':
      return 'bad'
    case 'cancelled':
      return 'mute'
    case 'running':
    case 'warmup':
    case 'testing':
    case 'connect':
      return 'run'
    default:
      return 'mute'
  }
}
