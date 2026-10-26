import type { ParamRegistry, ProbeResponse, RunDoc, RunSummary } from './types'

async function handle<T>(res: Response): Promise<T> {
  const text = await res.text()
  let body: any = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = text
  }
  if (!res.ok) {
    const detail =
      (body && typeof body === 'object' && (body.detail || body.message)) ||
      (typeof body === 'string' ? body : res.statusText)
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return body as T
}

export async function apiHealth(): Promise<{ ok: boolean; runs_total: number }> {
  return handle(await fetch('/api/health'))
}

export async function listPresets(): Promise<{ presets: Record<string, any> }> {
  return handle(await fetch('/api/presets'))
}

/** Parameter registry — the UI renders labels/units/tooltips from this. */
export async function getParams(): Promise<ParamRegistry> {
  return handle(await fetch('/api/params'))
}

export async function probe(baseUrl: string, apiKey: string): Promise<ProbeResponse> {
  return handle(
    await fetch('/api/probe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey || null }),
    })
  )
}

export interface CreateRunPayload {
  base_url: string
  api_key?: string | null
  model?: string | null
  preset: string
  label?: string | null
  /** pin the adapter instead of auto-detecting */
  family?: string | null
  /** transport-level knobs, see registry category `execution` */
  timeout_s?: number | null
  retries?: number | null
  /** task-level knobs, see registry categories input/output/sampling/execution */
  performance?: Record<string, any>
  /** context-task knobs (ladder / fill strategy / max-context detection) */
  context?: Record<string, any>
  /** concurrency-task knobs (ladder / requests per level) */
  concurrency?: Record<string, any>
}

export async function createRun(
  payload: CreateRunPayload
): Promise<{ run_id: string; status: string; note?: string | null }> {
  return handle(
    await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  )
}

export async function getRun(
  runId: string
): Promise<{ status: string; live: boolean; result: RunDoc }> {
  return handle(await fetch(`/api/runs/${runId}`))
}

export async function listRuns(): Promise<{ runs: RunSummary[]; total: number }> {
  return handle(await fetch('/api/runs?limit=50'))
}

export async function deleteRun(runId: string): Promise<{ ok: boolean }> {
  return handle(await fetch(`/api/runs/${runId}`, { method: 'DELETE' }))
}

export function openRunEvents(runId: string): EventSource {
  return new EventSource(`/api/runs/${runId}/events`)
}
