import { useCallback, useEffect, useRef, useState } from "react"
import {
  ResponsiveContainer, PieChart, Pie, Cell,
  AreaChart, Area, XAxis, YAxis, BarChart, Bar,
} from "recharts"
import { Sun, Moon, Upload, RefreshCw, ArrowRight, Check, ChevronDown } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import * as api from "@/lib/api"

// data colors from the theme's chart palette (chart-5 blue, chart-4 red)
const BLUE = "#0099FF"    // NORMAL
const ORANGE = "#FF4D50"  // PNEUMONIA
const pct = (v?: number) => (v == null ? "—" : (v * 100).toFixed(2) + "%")
const fmtUptime = (s: number) => {
  s = Math.max(0, Math.floor(s))
  const d = (s / 86400) | 0, h = (s / 3600) % 24 | 0, m = (s / 60) % 60 | 0, x = s % 60
  return `${d}d ${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m ${String(x).padStart(2, "0")}s`
}

/* ---------- primitives ---------- */
function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] uppercase tracking-[0.12em] font-heading opacity-70">{children}</div>
}
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xl font-heading mb-4 flex items-center gap-3">
      <span className="w-2.5 h-6 bg-main border-2 border-border rounded-[3px]" />
      {children}
    </h2>
  )
}

function Kpi({ label, value, className }: { label: string; value?: number; className?: string }) {
  return (
    <Card className={cn("gap-1 py-4 justify-center", className)}>
      <CardContent className="px-4">
        <Eyebrow>{label}</Eyebrow>
        <div className="font-heading text-2xl md:text-3xl my-1.5 tabular-nums">{pct(value)}</div>
        <Progress value={value == null ? 0 : value * 100} className="h-2.5" />
      </CardContent>
    </Card>
  )
}

function HeroMetric({ value }: { value?: number }) {
  return (
    <Card className="col-span-2 lg:col-span-4 gap-0 py-5">
      <CardContent className="flex flex-col h-full px-5">
        <div className="flex items-center justify-between">
          <Eyebrow>Held-out accuracy</Eyebrow>
          <Badge variant="neutral" className="text-[10px] px-2 py-0.5">624-image test</Badge>
        </div>
        <div className="font-heading text-5xl md:text-6xl leading-none my-4 tabular-nums">{pct(value)}</div>
        <Progress value={value == null ? 0 : value * 100} className="h-4 mt-auto" />
        <p className="text-xs opacity-70 mt-2.5">Beats the 62.5% majority-class baseline by a real margin.</p>
      </CardContent>
    </Card>
  )
}

/* collapsible interpretation used across the visualization cards */
function Interpretation({ takeaway, children }: { takeaway: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <p className="font-heading text-sm leading-snug">{takeaway}</p>
      <Collapsible>
        <CollapsibleTrigger className="group mt-2 inline-flex items-center gap-1.5 text-xs font-heading opacity-70 hover:opacity-100 cursor-pointer">
          <ChevronDown className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-180" />
          Full interpretation
        </CollapsibleTrigger>
        <CollapsibleContent className="text-sm opacity-80 pt-2 leading-relaxed">{children}</CollapsibleContent>
      </Collapsible>
    </div>
  )
}

function Confusion({ cm }: { cm: number[][] }) {
  const [[tn, fp], [fn, tp]] = cm
  const cells: [string, number, boolean][] = [["TN", tn, true], ["FP", fp, false], ["FN", fn, false], ["TP", tp, true]]
  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map(([lab, v, correct]) => (
        <div key={lab} className={cn("border-2 border-border rounded-base p-3 text-center shadow-shadow",
          correct ? "bg-main text-main-foreground" : "bg-background")}>
          <div className="font-heading text-2xl tabular-nums">{v}</div>
          <div className="text-[11px] opacity-80">{lab}</div>
        </div>
      ))}
    </div>
  )
}

function Donut({ normal, pneu }: { normal: number; pneu: number }) {
  const total = normal + pneu
  const data = [{ name: "NORMAL", value: normal, color: BLUE }, { name: "PNEUMONIA", value: pneu, color: ORANGE }]
  return (
    <div className="relative h-[150px]">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" innerRadius={46} outerRadius={68} paddingAngle={2}
            stroke="var(--border)" strokeWidth={2} startAngle={90} endAngle={-270} isAnimationActive={false}>
            {data.map((d) => <Cell key={d.name} fill={d.color} />)}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div className="font-heading text-2xl tabular-nums">{total.toLocaleString()}</div>
        <div className="text-[11px] opacity-70">images</div>
      </div>
    </div>
  )
}

function Brightness({ c }: { c: Record<string, api.ClassStat> }) {
  const data = c.NORMAL.brightness_hist.map((_, i) => ({
    bin: Math.round((i / 19) * 255), NORMAL: c.NORMAL.brightness_hist[i], PNEUMONIA: c.PNEUMONIA.brightness_hist[i],
  }))
  return (
    <ResponsiveContainer width="100%" height={150}>
      <AreaChart data={data} margin={{ top: 8, right: 6, left: 6, bottom: 0 }}>
        <XAxis dataKey="bin" type="number" domain={[0, 255]} ticks={[0, 128, 255]}
          tick={{ fontSize: 10, fill: "var(--foreground)" }} stroke="var(--border)" strokeWidth={2} />
        <YAxis hide />
        <Area type="monotone" dataKey="NORMAL" stroke={BLUE} strokeWidth={2} fill={BLUE} fillOpacity={0.18} isAnimationActive={false} />
        <Area type="monotone" dataKey="PNEUMONIA" stroke={ORANGE} strokeWidth={2} fill={ORANGE} fillOpacity={0.18} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function Contrast({ c }: { c: Record<string, api.ClassStat> }) {
  const data = [
    { name: "NORMAL", v: c.NORMAL.mean_contrast, color: BLUE },
    { name: "PNEUMONIA", v: c.PNEUMONIA.mean_contrast, color: ORANGE },
  ]
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 34, left: 6, bottom: 4 }}>
        <XAxis type="number" hide domain={[0, "dataMax + 12"]} />
        <YAxis type="category" dataKey="name" width={82} tick={{ fontSize: 10, fill: "var(--foreground)" }} stroke="var(--border)" strokeWidth={2} />
        <Bar dataKey="v" stroke="var(--border)" strokeWidth={2} barSize={30} isAnimationActive={false}
          label={{ position: "right", fontSize: 13, fontWeight: 700, fill: "var(--foreground)", formatter: (v: number) => v.toFixed(1) }}>
          {data.map((d) => <Cell key={d.name} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function MeanImage({ arr, label, color }: { arr?: number[][]; label: string; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv || !arr) return
    const ctx = cv.getContext("2d")!
    const im = ctx.createImageData(64, 64)
    arr.forEach((row, y) => row.forEach((p, x) => { const i = (y * 64 + x) * 4; im.data[i] = im.data[i + 1] = im.data[i + 2] = p; im.data[i + 3] = 255 }))
    ctx.putImageData(im, 0, 0)
  }, [arr])
  return (
    <div className="text-center">
      <canvas ref={ref} width={64} height={64}
        className="w-32 h-32 border-2 border-border rounded-base shadow-shadow mx-auto"
        style={{ imageRendering: "pixelated" }}
        role="img" aria-label={`Averaged ${label} chest X-ray`} />
      <div className="mt-2 text-xs font-heading" style={{ color }}>{label}</div>
    </div>
  )
}

function Row({ k, v, mono, small }: { k: string; v: string; mono?: boolean; small?: boolean }) {
  return (
    <div className="flex justify-between gap-3 border-b-2 border-border pb-2.5 last:border-0 last:pb-0">
      <span className="opacity-70 text-sm shrink-0">{k}</span>
      <span className={cn("text-right font-heading", mono && "font-mono text-sm", small && "text-xs font-base max-w-[62%]")}>{v}</span>
    </div>
  )
}

/* ---------- app ---------- */
export default function App() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"))
  const toggleTheme = () => {
    const d = !dark
    document.documentElement.classList.toggle("dark", d)
    localStorage.setItem("theme", d ? "dark" : "light")
    setDark(d)
  }

  const [status, setStatus] = useState<api.Status | null>(null)
  const [online, setOnline] = useState(true)
  const [uptime, setUptime] = useState("—")
  const baseRef = useRef(0), atRef = useRef(0)
  const loadStatus = useCallback(async () => {
    try {
      const s = await api.getStatus()
      setStatus(s); setOnline(true); baseRef.current = s.uptime_seconds; atRef.current = performance.now()
    } catch { setOnline(false) }
  }, [])
  useEffect(() => {
    loadStatus()
    const iv = setInterval(loadStatus, 5000)
    const tick = setInterval(() => { if (atRef.current) setUptime(fmtUptime(baseRef.current + (performance.now() - atRef.current) / 1000)) }, 1000)
    return () => { clearInterval(iv); clearInterval(tick) }
  }, [loadStatus])

  const [viz, setViz] = useState<api.Viz | null>(null)
  useEffect(() => { api.getViz().then(setViz).catch(() => {}) }, [])

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState("")
  const [pred, setPred] = useState<api.Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [predErr, setPredErr] = useState("")
  const onPickPredict = (f?: File) => { if (!f) return; setFile(f); setPred(null); setPredErr(""); setPreview(URL.createObjectURL(f)) }
  const doPredict = async () => {
    if (!file) return
    setPredicting(true); setPredErr("")
    try { setPred(await api.predict(file)) } catch (e: any) { setPredErr(e.message) }
    setPredicting(false)
  }

  const [label, setLabel] = useState("NORMAL")
  const [upFiles, setUpFiles] = useState<FileList | null>(null)
  const [upMsg, setUpMsg] = useState("")
  const [uploading, setUploading] = useState(false)
  const doUpload = async () => {
    if (!upFiles?.length) return
    setUploading(true)
    try { const r: any = await api.upload(upFiles, label); setUpMsg(`Saved ${r.saved} image(s) as ${r.label}. Pool: ${r.total_pending_upload_images} pending.`) }
    catch (e: any) { setUpMsg("Error: " + e.message) }
    setUploading(false)
  }

  const [retrainMsg, setRetrainMsg] = useState("Idle.")
  const [retraining, setRetraining] = useState(false)
  const [retrainDone, setRetrainDone] = useState(false)
  const poll = useCallback(async () => {
    try {
      const s = await api.retrainStatus()
      if (s.status === "running") { setRetrainMsg(`Fine-tuning & re-evaluating… (started ${s.started_at})`); setTimeout(poll, 3000); return }
      if (s.status === "done") {
        const acc = s.detail?.metrics?.accuracy, n = s.detail?.n_new_uploaded
        setRetrainMsg(`Done. Hot-swapped into the live API. Trained on base + ${n || 0} uploaded.` + (acc != null ? ` New test accuracy ${(acc * 100).toFixed(2)}%.` : ""))
        setRetrainDone(true); loadStatus()
      } else { setRetrainMsg(`Status: ${s.status}`) }
      setRetraining(false)
    } catch (e: any) { setRetrainMsg("Error: " + e.message); setRetraining(false) }
  }, [loadStatus])
  const doRetrain = async () => {
    setRetraining(true); setRetrainDone(false)
    try { const r: any = await api.retrain(); setRetrainMsg(r.message || "Retraining started…"); poll() }
    catch (e: any) { setRetrainMsg("Error: " + e.message); setRetraining(false) }
  }

  const m = status?.model_metrics || {}
  const c = viz?.classes

  return (
    <div className="min-h-screen gridbg text-foreground font-base">
      <div className="max-w-[1280px] mx-auto px-5 py-8 space-y-8">

        {/* header */}
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 grid place-items-center text-2xl border-2 border-border rounded-base bg-main shadow-shadow">🫁</div>
            <div>
              <h1 className="text-3xl md:text-4xl font-heading leading-none">Pneumonia Detection</h1>
              <p className="text-sm mt-1.5 opacity-80">Chest X-ray classifier · live model dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge className={cn("border-2 border-border shadow-shadow gap-2 px-3 py-1.5 text-sm",
              online ? "bg-[oklch(79.76%_0.2044_153.08)] text-black" : "bg-[oklch(67.28%_0.2147_24.22)] text-white")}>
              <span className="w-2 h-2 rounded-full bg-current" />{online ? "Operational" : "Offline"}
            </Badge>
            <Button variant="neutral" size="icon" onClick={toggleTheme} aria-label="Toggle theme">{dark ? <Sun /> : <Moon />}</Button>
          </div>
        </header>

        {/* ===== BENTO BAND 1 — hero: predict + headline metrics ===== */}
        <section className="grid grid-cols-2 lg:grid-cols-12 gap-4 lg:auto-rows-fr">
          {/* Predict = the hero */}
          <Card className="col-span-2 lg:col-span-5 lg:row-span-2">
            <CardHeader><CardTitle className="text-xl">Predict a chest X-ray</CardTitle></CardHeader>
            <CardContent className="flex-1 flex flex-col gap-3">
              <label className="flex-1 flex flex-col items-center justify-center gap-2 border-2 border-dashed border-border rounded-base bg-background p-8 text-center cursor-pointer hover:bg-main/10 min-h-[150px]">
                {preview
                  ? <img src={preview} alt="preview" className="w-24 h-24 object-cover border-2 border-border rounded-base" />
                  : <Upload className="opacity-70 size-7" />}
                <span className="text-sm opacity-80">{file ? file.name : "Drop an X-ray here, or click to browse (PNG / JPEG)"}</span>
                <input type="file" accept="image/*" className="hidden" onChange={(e) => onPickPredict(e.target.files?.[0])} />
              </label>
              <Button className="w-full h-12 text-base" disabled={!file || predicting} onClick={doPredict}>
                {predicting ? "Analyzing…" : "Predict"}<ArrowRight />
              </Button>
              {predErr && <div className="text-sm font-heading text-[oklch(67.28%_0.2147_24.22)]">{predErr}</div>}
              {pred && (
                <div className="border-2 border-border rounded-base p-4 bg-background space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-heading text-2xl" style={{ color: pred.prediction === "PNEUMONIA" ? ORANGE : BLUE }}>{pred.prediction}</span>
                    <Badge variant="neutral">{(pred.confidence * 100).toFixed(1)}% confidence</Badge>
                  </div>
                  <div className="flex justify-between text-xs opacity-70"><span>pneumonia probability</span><span className="tabular-nums">{(pred.pneumonia_probability * 100).toFixed(1)}%</span></div>
                  <Progress value={pred.pneumonia_probability * 100} />
                  {(pred.threshold ?? status?.decision_threshold ?? m.decision_threshold) != null && (
                    <div className="flex justify-between text-xs opacity-70">
                      <span>decision threshold</span>
                      <span className="tabular-nums">{pred.threshold ?? status?.decision_threshold ?? m.decision_threshold}</span>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <HeroMetric value={m.accuracy} />
          <Kpi label="ROC AUC" value={m.roc_auc} className="col-span-1 lg:col-span-3" />
          <Kpi label="Precision" value={m.precision} className="col-span-1 lg:col-span-2" />
          <Kpi label="Recall" value={m.recall} className="col-span-1 lg:col-span-2" />
          <Kpi label="F1 score" value={m.f1_score} className="col-span-1 lg:col-span-3" />
        </section>

        {/* ===== BENTO BAND 2 — production health + confusion matrix ===== */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <Card className="lg:col-span-7">
            <CardHeader><CardTitle className="text-lg">Model &amp; production health</CardTitle></CardHeader>
            <CardContent className="space-y-2.5">
              <Row k="Status" v={online ? "Operational" : "Offline"} />
              <Row k="Uptime" v={uptime} mono />
              <Row k="Last trained" v={(status?.model_saved_at || "—").replace(" ", " · ")} mono />
              <Row k="Version" v={status?.model_note || "—"} small />
              <Row k="Decision threshold" v={String(status?.decision_threshold ?? m.decision_threshold ?? "—")} mono />
              <Row k="Test set" v={m.n_test ? `${m.n_test} images` : "—"} />
            </CardContent>
          </Card>
          <Card className="lg:col-span-5">
            <CardHeader><CardTitle className="text-lg">Confusion matrix</CardTitle></CardHeader>
            <CardContent>
              {m.confusion_matrix ? <Confusion cm={m.confusion_matrix} /> : <div className="text-sm opacity-60">loading…</div>}
              <p className="text-xs opacity-70 mt-3">Rows = actual, columns = predicted (NORMAL, PNEUMONIA). Blue cells are correct predictions on the held-out test set.</p>
            </CardContent>
          </Card>
        </section>

        {/* ===== BENTO BAND 3 — dataset visualizations & interpretations ===== */}
        <section>
          <SectionLabel>Dataset visualizations &amp; interpretations</SectionLabel>
          {c ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-4">
              {/* Feature 1 */}
              <Card className="lg:col-span-4">
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between"><Eyebrow>Feature 01 · Class balance</Eyebrow><Badge variant="neutral" className="text-[10px] px-2 py-0.5">bar = 62.5%</Badge></div>
                  <Donut normal={c.NORMAL.count} pneu={c.PNEUMONIA.count} />
                  <div className="flex gap-4 text-xs">
                    <span className="flex items-center gap-1.5"><i className="w-3 h-3 border border-border" style={{ background: BLUE }} />NORMAL {c.NORMAL.count.toLocaleString()}</span>
                    <span className="flex items-center gap-1.5"><i className="w-3 h-3 border border-border" style={{ background: ORANGE }} />PNEUMONIA {c.PNEUMONIA.count.toLocaleString()}</span>
                  </div>
                  <Interpretation takeaway={`${(c.PNEUMONIA.count / (c.NORMAL.count + c.PNEUMONIA.count) * 100).toFixed(1)}% of the set is PNEUMONIA, so 62.5% (not 50%) is the accuracy bar.`}>
                    A model that always guessed PNEUMONIA would already score 62.5% on the test set, so that, not 50%, is the number any accuracy claim has to clear. Training applies class weights (NORMAL ×1.94, PNEUMONIA ×0.67) to offset the imbalance so the rarer NORMAL class is not ignored.
                  </Interpretation>
                </CardContent>
              </Card>

              {/* Feature 2 (wide) */}
              <Card className="lg:col-span-5">
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between"><Eyebrow>Feature 02 · Pixel brightness</Eyebrow><Badge variant="neutral" className="text-[10px] px-2 py-0.5">negative result</Badge></div>
                  <Brightness c={c} />
                  <div className="flex gap-4 text-xs">
                    <span className="flex items-center gap-1.5"><i className="w-3 h-3 border border-border" style={{ background: BLUE }} />NORMAL μ {c.NORMAL.mean_brightness.toFixed(1)}</span>
                    <span className="flex items-center gap-1.5"><i className="w-3 h-3 border border-border" style={{ background: ORANGE }} />PNEUMONIA μ {c.PNEUMONIA.mean_brightness.toFixed(1)}</span>
                  </div>
                  <Interpretation takeaway="Pneumonia is not brighter on average: a plausible feature that carries no signal (Cohen's d = 0.04).">
                    Intuition says infection should show as brighter white opacity. It does not hold: the means are {c.NORMAL.mean_brightness.toFixed(1)} vs {c.PNEUMONIA.mean_brightness.toFixed(1)} on a 0 to 255 scale, statistically indistinguishable, and the two curves sit almost on top of each other. Pneumonia is a localised opacity that averaging every pixel washes out, which is exactly why a CNN that reads local spatial pattern beats a global statistic.
                  </Interpretation>
                </CardContent>
              </Card>

              {/* Feature 3 */}
              <Card className="lg:col-span-3">
                <CardContent className="space-y-3">
                  <Eyebrow>Feature 03 · Contrast</Eyebrow>
                  <Contrast c={c} />
                  <Interpretation takeaway="Contrast is where the signal lives: consolidation compresses the pixel spread (d = -0.62).">
                    Healthy lungs are air-filled and near-black against bright ribs, giving a wide spread of pixel values (NORMAL {c.NORMAL.mean_contrast.toFixed(1)}). Pneumonic consolidation floods those dark spaces with white opacity and compresses the spread (PNEUMONIA {c.PNEUMONIA.mean_contrast.toFixed(1)}). A real medium effect, and it survives averaging because it changes the spread, not the mean.
                  </Interpretation>
                </CardContent>
              </Card>

              {/* Feature 4 (full width, images + text) */}
              <Card className="lg:col-span-12">
                <CardContent className="flex flex-col md:flex-row gap-6 items-start">
                  <div className="flex gap-4 shrink-0">
                    <MeanImage arr={c.NORMAL.mean_image} label="NORMAL" color={BLUE} />
                    <MeanImage arr={c.PNEUMONIA.mean_image} label="PNEUMONIA" color={ORANGE} />
                  </div>
                  <div className="space-y-3">
                    <Eyebrow>Feature 04 · Mean image per class</Eyebrow>
                    <Interpretation takeaway="Averaged, pneumonia reads as hazier, lower-contrast lung fields: a spatial signature, not a single number.">
                      Averaging every image per class cancels individual anatomy and leaves the population fingerprint: the NORMAL mean keeps darker, sharper lung fields, while the PNEUMONIA mean is hazier and lower in contrast, in the lung zones specifically. A single number per image cannot capture where the change is, which is the whole case for a convolutional model over hand-crafted statistics.
                    </Interpretation>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : <div className="opacity-60">loading visualizations…</div>}
        </section>

        {/* ===== BENTO BAND 4 — actions ===== */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">Upload data for retraining</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Select value={label} onValueChange={setLabel}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="NORMAL">Label: NORMAL</SelectItem>
                  <SelectItem value="PNEUMONIA">Label: PNEUMONIA</SelectItem>
                </SelectContent>
              </Select>
              <label className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-border rounded-base bg-background p-6 text-center cursor-pointer hover:bg-main/10">
                <Upload className="opacity-70" />
                <span className="text-sm opacity-80">{upFiles?.length ? `${upFiles.length} image(s) ready` : "Drop many labelled X-rays here, or click to browse"}</span>
                <input type="file" accept="image/*" multiple className="hidden" onChange={(e) => setUpFiles(e.target.files)} />
              </label>
              <Button variant="neutral" className="w-full" disabled={!upFiles?.length || uploading} onClick={doUpload}>{uploading ? "Uploading…" : "Upload images"}</Button>
              {upMsg && <p className="text-sm">{upMsg}</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-lg">Retrain the model</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm opacity-80">Fine-tunes the saved model as a pre-trained base on the original training set plus every uploaded image, re-evaluates on the held-out test set, then hot-swaps the new model into the live API.</p>
              <Button className="w-full" disabled={retraining} onClick={doRetrain}>
                <RefreshCw className={retraining ? "animate-spin" : ""} />{retraining ? "Retraining…" : "Retrain model"}
              </Button>
              <p className="text-sm flex items-start gap-2">
                {retrainDone && <Check className="text-[oklch(79.76%_0.2044_153.08)] shrink-0 mt-0.5" />}
                <span>{retrainMsg}</span>
              </p>
            </CardContent>
          </Card>
        </section>

        <footer className="pt-5 border-t-2 border-border text-xs opacity-70">
          Pediatric chest X-rays (Kermany et al., Guangzhou) · CC BY 4.0. A pipeline-engineering project, not a diagnostic tool.
        </footer>
      </div>
    </div>
  )
}
