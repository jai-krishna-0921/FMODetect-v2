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

export default function Home() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [image, setImage] = useState<File | null>(null);
  const [bg, setBg] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [result, setResult] = useState<ImageResult | VideoResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetch("/api/info").then(r => r.json()).then(setInfo).catch(() => {});
  }, []);

  const submit = async () => {
    setBusy(true); setErr(null); setResult(null);
    const fd = new FormData();
    try {
      let path = "";
      if (mode === "image") {
        if (!image || !bg) throw new Error("upload both image and background");
        fd.append("image", image); fd.append("background", bg);
        path = "/api/infer/image";
      } else {
        if (!video) throw new Error("upload a video");
        fd.append("video", video);
        path = "/api/infer/video";
      }
      const r = await fetch(path, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`api error ${r.status}: ${await r.text()}`);
      setResult(await r.json());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <header className="mb-10">
        <h1 className="text-4xl font-bold tracking-tight">FMODetect<span className="text-orange-400">·v2</span></h1>
        <p className="mt-2 text-neutral-400">
          CBAM attention · joint TDF + matting · uncertainty-weighted boundary loss
        </p>
        {info && (
          <div className="mt-3 text-xs text-neutral-500">
            device: <code>{String(info.device)}</code> · gpu: <code>{String(info.gpu ?? "—")}</code> ·
            ckpt: <code>{String(info.ckpt ?? "(none)")}</code>
          </div>
        )}
      </header>

      <div className="mb-6 flex gap-2">
        {(["image", "video"] as const).map(m => (
          <button
            key={m}
            onClick={() => { setMode(m); setResult(null); setErr(null); }}
            className={`rounded-md px-4 py-2 text-sm font-medium ${
              mode === m ? "bg-orange-500 text-black" : "bg-neutral-900 text-neutral-300 hover:bg-neutral-800"
            }`}
          >{m.toUpperCase()}</button>
        ))}
      </div>

      <section className="rounded-lg border border-neutral-800 bg-neutral-950 p-6">
        {mode === "image" ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <FileSlot label="Image with FMO" accept="image/*" file={image} onChange={setImage} />
            <FileSlot label="Background" accept="image/*" file={bg} onChange={setBg} />
          </div>
        ) : (
          <FileSlot label="Video clip" accept="video/*" file={video} onChange={setVideo} />
        )}
        <button
          onClick={submit}
          disabled={busy}
          className="mt-6 rounded-md bg-orange-500 px-6 py-2.5 font-semibold text-black disabled:opacity-50"
        >{busy ? "Running…" : "Detect FMO"}</button>
        {err && <p className="mt-3 text-sm text-red-400">{err}</p>}
      </section>

      {result && (
        <section className="mt-10">
          <h2 className="mb-4 text-lg font-semibold">Detection</h2>
          {"overlay_url" in result && result.overlay_url.endsWith(".mp4") ? (
            <video controls className="w-full rounded-lg border border-neutral-800" src={result.overlay_url} />
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <figure>
                  <figcaption className="mb-1 text-xs text-neutral-400">
                    Overlay ({(result as ImageResult).n_detections} detection{(result as ImageResult).n_detections !== 1 ? "s" : ""})
                  </figcaption>
                  <img className="w-full rounded border border-neutral-800" src={(result as ImageResult).overlay_url} />
                </figure>
                <figure>
                  <figcaption className="mb-1 text-xs text-neutral-400">Truncated Distance Function</figcaption>
                  <img className="w-full rounded border border-neutral-800" src={(result as ImageResult).tdf_url} />
                </figure>
              </div>
              {(result as ImageResult).trajectories?.length > 0 && (
                <div className="mt-6">
                  <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">
                    Trajectories
                  </h3>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {(result as ImageResult).trajectories.map((t, i) => (
                      <div key={i} className="rounded border border-neutral-800 bg-neutral-900 p-3 text-xs">
                        <div className="mb-1 font-mono text-orange-400">FMO #{i + 1}</div>
                        <Row k="length" v={`${t.length_px.toFixed(1)} px`} />
                        <Row k="speed" v={`${t.speed_px_per_frame.toFixed(1)} px / frame`} />
                        <Row k="radius" v={`${t.radius_px.toFixed(1)} px`} />
                        <Row k="confidence" v={(t.confidence * 100).toFixed(1) + "%"} />
                        <Row k="pixels" v={String(t.n_pixels)} />
                        <Row k="bbox (yxyx)" v={t.bbox_yxyx.join(", ")} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}
    </main>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-neutral-500">{k}</span>
      <span className="text-neutral-200">{v}</span>
    </div>
  );
}

function FileSlot({ label, accept, file, onChange }:
  { label: string; accept: string; file: File | null; onChange: (f: File | null) => void }) {
  return (
    <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed
                      border-neutral-700 bg-neutral-900 p-6 text-center text-sm hover:border-orange-400">
      <span className="text-neutral-400">{label}</span>
      <span className="mt-2 text-neutral-200">{file ? file.name : "click or drop file"}</span>
      <input type="file" accept={accept} className="hidden"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
    </label>
  );
}
