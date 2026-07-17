// Thin API layer. Same FastAPI endpoints the old UI used.
export type Metrics = {
  accuracy: number; precision: number; recall: number; f1_score: number;
  roc_auc: number; confusion_matrix: number[][]; n_test: number
}
export type Status = {
  status: string; uptime_seconds: number; uptime_human: string;
  model_saved_at: string | null; model_note: string | null;
  model_metrics: Partial<Metrics>; retraining: string
}
export type ClassStat = {
  count: number; mean_brightness: number; mean_contrast: number;
  brightness_hist: number[]; mean_image: number[][]
}
export type Viz = { classes: Record<string, ClassStat> }
export type Prediction = {
  prediction: string
  pneumonia_probability: number
  confidence: number
  threshold?: number
  filename?: string
}
export type RetrainState = { status: string; detail: any; started_at: string | null; finished_at: string | null }

async function jfetch(url: string, opts?: RequestInit) {
  const r = await fetch(url, opts)
  if (!r.ok) {
    let d: any = {}
    try { d = await r.json() } catch { /* ignore */ }
    throw new Error(d.detail || `HTTP ${r.status}`)
  }
  return r.json()
}

export const getStatus = () => jfetch('/status') as Promise<Status>
export const getViz = () => jfetch('/visualizations') as Promise<Viz>
export const predict = (f: File) => {
  const fd = new FormData(); fd.append('file', f)
  return jfetch('/predict', { method: 'POST', body: fd }) as Promise<Prediction>
}
export const upload = (files: FileList, label: string) => {
  const fd = new FormData()
  for (const f of Array.from(files)) fd.append('files', f)
  fd.append('label', label)
  return jfetch('/upload', { method: 'POST', body: fd })
}
export const retrain = () => jfetch('/retrain', { method: 'POST' })
export const retrainStatus = () => jfetch('/retrain/status') as Promise<RetrainState>
