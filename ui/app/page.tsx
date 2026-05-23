"use client";
import { useEffect, useState } from "react";

type Trajectory = {
  length_px: number;
  speed_px_per_frame: number;
  radius_px: number;
  confidence: number;
  n_pixels: number;
  bbox_yxyx: [number, number, number, number];
  is_bounce: boolean;
};
type ImageResult = { overlay_url: string; tdf_url: string; trajectories: Trajectory[]; n_detections: number };
type VideoResult = { overlay_url: string; frames: number; out: string };
type Example = { name: string; label: string; thumb_url: string };

export default function Home() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [image, setImage] = useState<File | null>(null);
  const [bg, setBg] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [result, setResult] = useState<ImageResult | VideoResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [activeSample, setActiveSample] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/info").then(r => r.json()).then(setInfo).catch(() => {});
    fetch("/api/examples").then(r => r.json()).then(setExamples).catch(() => setExamples([]));
  }, []);

  const runSample = async (name: string) => {
    setBusy(true); setErr(null); setResult(null); setActiveSample(name);
    setMode("image");
    try {
      const r = await fetch(`/api/infer/sample/${name}`, { method: "POST" });
      if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
      setResult(await r.json());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    setBusy(true); setErr(null); setResult(null); setActiveSample(null);
    const fd = new FormData();
    try {
      let path = "";
      if (mode === "image") {
        if (!image || !bg) throw new Error("Please provide both an image and a background.");
        fd.append("image", image); fd.append("background", bg);
        path = "/api/infer/image";
      } else {
        if (!video) throw new Error("Please provide a video clip.");
        fd.append("video", video);
        path = "/api/infer/video";
      }
      const r = await fetch(path, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
      setResult(await r.json());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto max-w-3xl px-8 py-20">
      <header className="mb-16">
        <div className="mono mb-3 text-[11px] uppercase tracking-[0.18em] text-[--text-muted]">
          Research demo · v0.2
        </div>
        <h1 className="serif text-[2.5rem] font-normal leading-tight tracking-tight text-[--text]">
          Detecting fast-moving objects<br />
          <span className="italic text-[--text-soft]">from a single blurred frame.</span>
        </h1>
        <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-[--text-soft]">
          A PyTorch re-implementation of FMODetect with three additions:
          CBAM attention, a joint truncated-distance / matting head, and an
          uncertainty-weighted boundary loss.
        </p>
        {info && (
          <div className="mono mt-6 flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-[--text-muted]">
            <span>device <span className="text-[--text-soft]">{String(info.device ?? "—")}</span></span>
            <span>gpu <span className="text-[--text-soft]">{String(info.gpu ?? "cpu")}</span></span>
            <span>checkpoint <span className="text-[--text-soft]">{abbrev(info.ckpt)}</span></span>
          </div>
        )}
      </header>

      {examples.length > 0 && (
        <div className="mb-10">
          <div className="mono mb-3 text-[10px] uppercase tracking-[0.18em] text-[--text-muted]">
            Try a sample
          </div>
          <div className="flex flex-wrap gap-3">
            {examples.map(ex => (
              <button
                key={ex.name}
                onClick={() => runSample(ex.name)}
                disabled={busy}
                className={`group relative overflow-hidden border transition-colors disabled:opacity-50 ${
                  activeSample === ex.name && result
                    ? "border-[--accent]"
                    : "border-[--border] hover:border-[--border-strong]"
                }`}
              >
                <img src={ex.thumb_url} alt={ex.label} className="block h-20 w-32 object-cover opacity-90 transition-opacity group-hover:opacity-100" />
                <span className="mono absolute bottom-0 left-0 right-0 bg-black/55 px-2 py-1 text-left text-[10px] uppercase tracking-wider text-[--text-soft]">
                  {ex.label}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mb-8 flex items-center gap-8 border-b border-[--border] pb-3">
        {(["image", "video"] as const).map(m => (
          <button
            key={m}
            onClick={() => { setMode(m); setResult(null); setErr(null); }}
            className={`relative text-[13px] tracking-wide transition-colors ${
              mode === m ? "text-[--text]" : "text-[--text-muted] hover:text-[--text-soft]"
            }`}
          >
            {m === "image" ? "Image pair" : "Video clip"}
            {mode === m && (
              <span className="absolute -bottom-[14px] left-0 right-0 h-px bg-[--accent]" />
            )}
          </button>
        ))}
      </div>

      <section>
        {mode === "image" ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <FileSlot label="Frame with FMO" hint="single image, motion-blurred subject" accept="image/*" file={image} onChange={setImage} />
            <FileSlot label="Background" hint="same scene without the object" accept="image/*" file={bg} onChange={setBg} />
          </div>
        ) : (
          <FileSlot label="Video clip" hint="mp4 or mov, any length" accept="video/*" file={video} onChange={setVideo} />
        )}

        <div className="mt-8 flex items-center gap-5">
          <button
            onClick={submit}
            disabled={busy}
            className="group inline-flex items-center gap-2 border border-[--accent-soft] px-5 py-2 text-[13px] tracking-wide text-[--accent] transition-all hover:bg-[--accent-soft]/15 disabled:opacity-40"
          >
            {busy ? (
              <>
                <span className="inline-block h-1 w-1 animate-pulse rounded-full bg-[--accent]" />
                Running inference…
              </>
            ) : (
              <>Detect <span className="text-[--accent-soft] transition-colors group-hover:text-[--accent]">→</span></>
            )}
          </button>
          {err && <p className="text-[12px] text-rose-400/80">{err}</p>}
        </div>
      </section>

      {result && (
        <section className="mt-20">
          <div className="serif mb-6 text-[11px] uppercase tracking-[0.18em] text-[--text-muted]">
            <span className="not-italic">— Result</span>
          </div>

          {"overlay_url" in result && result.overlay_url.endsWith(".mp4") ? (
            <video
              controls
              className="w-full border border-[--border]"
              src={result.overlay_url}
            />
          ) : (
            <>
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                <Figure
                  caption={`Overlay — ${(result as ImageResult).n_detections} detection${
                    (result as ImageResult).n_detections === 1 ? "" : "s"
                  }`}
                  src={(result as ImageResult).overlay_url}
                />
                <Figure
                  caption="Truncated distance function"
                  src={(result as ImageResult).tdf_url}
                />
              </div>

              {(result as ImageResult).trajectories?.length > 0 && (
                <div className="mt-14">
                  <h3 className="serif mb-5 text-[11px] uppercase tracking-[0.18em] text-[--text-muted]">
                    Trajectories
                  </h3>
                  <div className="divide-y divide-[--border] border-y border-[--border]">
                    {(result as ImageResult).trajectories.map((t, i) => (
                      <div key={i} className="grid grid-cols-12 gap-4 py-4 text-[13px]">
                        <div className="mono col-span-2 text-[--accent]">#{String(i + 1).padStart(2, "0")}</div>
                        <Stat k="length"     v={`${t.length_px.toFixed(1)} px`} />
                        <Stat k="speed"      v={`${t.speed_px_per_frame.toFixed(1)} px/f`} />
                        <Stat k="radius"     v={`${t.radius_px.toFixed(1)} px`} />
                        <Stat k="confidence" v={(t.confidence * 100).toFixed(0) + "%"} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}

      <footer className="mono mt-24 border-t border-[--border] pt-6 text-[11px] text-[--text-muted]">
        After Rozumnyi et al., <span className="italic">FMODetect</span> (ICCV 2021) ·
        novelty notes in <span className="text-[--text-soft]">NOVELTY.md</span>
      </footer>
    </main>
  );
}

function Figure({ caption, src }: { caption: string; src: string }) {
  return (
    <figure>
      <img className="w-full border border-[--border]" src={src} alt={caption} />
      <figcaption className="mono mt-2 text-[11px] text-[--text-muted]">{caption}</figcaption>
    </figure>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="col-span-5 sm:col-span-2 sm:col-start-auto">
      <div className="mono text-[10px] uppercase tracking-wider text-[--text-muted]">{k}</div>
      <div className="mono mt-0.5 text-[--text-soft]">{v}</div>
    </div>
  );
}

function FileSlot({
  label, hint, accept, file, onChange,
}: {
  label: string; hint: string; accept: string;
  file: File | null; onChange: (f: File | null) => void;
}) {
  return (
    <label className="group block cursor-pointer border border-[--border] bg-[--surface] p-6 transition-colors hover:border-[--border-strong]">
      <div className="flex items-baseline justify-between">
        <span className="text-[13px] text-[--text]">{label}</span>
        <span className="mono text-[10px] uppercase tracking-wider text-[--text-muted]">
          {file ? "selected" : "empty"}
        </span>
      </div>
      <div className="mono mt-3 truncate text-[12px] text-[--text-soft]">
        {file ? file.name : <span className="text-[--text-muted]">{hint}</span>}
      </div>
      <input type="file" accept={accept} className="hidden"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
    </label>
  );
}

function abbrev(v: unknown): string {
  if (!v || typeof v !== "string") return "—";
  const parts = v.split("/");
  return parts.length > 2 ? "…/" + parts.slice(-2).join("/") : v;
}
