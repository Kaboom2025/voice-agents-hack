import { useEffect, useRef, useState } from "react";
import { api, type LiveFrame } from "../api";

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

function LiveTile({ cameraId }: { cameraId: string }) {
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const stopped = useRef(false);

  useEffect(() => {
    stopped.current = false;
    const ctrl = new AbortController();
    (async () => {
      for await (const f of api.liveStream([cameraId], ctrl.signal)) {
        if (stopped.current) break;
        setFrame(f);
      }
    })();
    return () => {
      stopped.current = true;
      ctrl.abort();
    };
  }, [cameraId]);

  return (
    <div className="relative aspect-video rounded-2xl overflow-hidden glass">
      <div className="absolute inset-0 grid place-items-center text-mute text-xs font-mono">
        {/* Replace with <video> wired to the real live transport. */}
        no signal
      </div>
      <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-ink/85 text-[11px] font-mono text-white">
        <span className="size-1.5 rounded-full bg-accent animate-pulse" />
        {cameraId}
      </div>
      {frame && (
        <div className="absolute bottom-3 right-3 px-2 py-0.5 rounded-full bg-ink/75 text-[10px] text-white/80 font-mono">
          {new Date(frame.ts_ms).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}
