import { useEffect, useRef, useState } from "react";
import { liveJpegUrl } from "../api";

function gridCols(n: number): string {
  if (n <= 1) return "grid-cols-1";
  if (n === 2) return "grid-cols-2";
  if (n <= 4) return "grid-cols-2";
  if (n <= 9) return "grid-cols-3";
  return "grid-cols-4";
}

export function LiveGrid({ cameraIds }: { cameraIds: string[] }) {
  if (cameraIds.length === 0) {
    return (
      <div className="glass rounded-3xl p-12 text-center text-mute text-sm">
        Pick a camera scope to begin streaming.
      </div>
    );
  }
  return (
    <div className={`grid ${gridCols(cameraIds.length)} gap-3`}>
      {cameraIds.map((id) => (
        <LiveTile key={id} cameraId={id} />
      ))}
    </div>
  );
}

// Self-driving image refresh: as soon as the previous JPEG decodes (load) or
// fails (error), schedule the next request via requestAnimationFrame. This
// caps the loop at the display refresh (~60fps) without queueing requests
// faster than the network can deliver.
function LiveTile({ cameraId }: { cameraId: string }) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [lastTs, setLastTs] = useState<number | null>(null);
  const [errored, setErrored] = useState(false);

  useEffect(() => {
    let alive = true;
    let raf = 0;

    const tick = () => {
      if (!alive || !imgRef.current) return;
      imgRef.current.src = liveJpegUrl(cameraId);
    };

    const onLoad = () => {
      if (!alive) return;
      setErrored(false);
      setLastTs(Date.now());
      raf = requestAnimationFrame(tick);
    };

    const onError = () => {
      if (!alive) return;
      setErrored(true);
      // Back off briefly on errors so we don't hammer the leader with
      // 404s when a follower is offline.
      raf = window.setTimeout(tick, 500) as unknown as number;
    };

    const img = imgRef.current;
    img?.addEventListener("load", onLoad);
    img?.addEventListener("error", onError);
    tick();

    return () => {
      alive = false;
      if (raf) {
        cancelAnimationFrame(raf);
        clearTimeout(raf);
      }
      img?.removeEventListener("load", onLoad);
      img?.removeEventListener("error", onError);
    };
  }, [cameraId]);

  return (
    <div className="relative aspect-video rounded-2xl overflow-hidden glass bg-slate-900">
      <img
        ref={imgRef}
        alt={cameraId}
        className="absolute inset-0 w-full h-full object-cover"
      />
      {(errored || lastTs === null) && (
        <div className="absolute inset-0 grid place-items-center text-mute text-xs font-mono pointer-events-none">
          {errored ? "no signal" : "connecting…"}
        </div>
      )}
      <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-ink/85 text-[11px] font-mono text-white">
        <span
          className={`size-1.5 rounded-full ${
            errored ? "bg-rose-400" : "bg-accent animate-pulse"
          }`}
        />
        {cameraId}
      </div>
      {lastTs !== null && !errored && (
        <div className="absolute bottom-3 right-3 px-2 py-0.5 rounded-full bg-ink/75 text-[10px] text-white/80 font-mono">
          {new Date(lastTs).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}
