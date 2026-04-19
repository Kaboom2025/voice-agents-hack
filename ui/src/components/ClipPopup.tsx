import { useEffect, useMemo, useState } from "react";
import { api, type ClipHit } from "../api";

interface ClipPopupProps {
  hits: ClipHit[];
  initialId?: string;
  onClose: () => void;
}

export function ClipPopup({ hits, initialId, onClose }: ClipPopupProps) {
  const ranked = useMemo(() => [...hits].sort((a, b) => b.score - a.score), [hits]);
  const [activeId, setActiveId] = useState<string>(
    initialId ?? ranked[0]?.chunk_id ?? "",
  );

  const active = ranked.find((h) => h.chunk_id === activeId) ?? ranked[0];
  const rest = ranked.filter((h) => h.chunk_id !== active?.chunk_id);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  if (!active) return null;

  const start = new Date(active.start_ts_ms);
  const scorePct = Math.round(active.score * 100);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl bg-white rounded-3xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-100">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
            {active.camera_id} · {start.toLocaleTimeString()} · {scorePct > 0 ? `${scorePct}% match` : "recent"}
          </div>
          <button
            onClick={onClose}
            className="size-7 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Main player */}
        <div className="bg-slate-950 aspect-video">
          <video
            key={active.chunk_id}
            src={api.clipUrl(active.chunk_id)}
            poster={active.thumbnail_url}
            controls
            autoPlay
            className="w-full h-full object-contain"
          />
        </div>

        {/* Caption */}
        {active.caption && (
          <div className="px-5 py-3 text-sm text-slate-700 border-b border-slate-100">
            {active.caption}
          </div>
        )}

        {/* Thumbnail strip */}
        {rest.length > 0 && (
          <div className="p-4">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-2">
              Other clips
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {rest.map((h) => {
                const s = new Date(h.start_ts_ms);
                const sp = Math.round(h.score * 100);
                return (
                  <button
                    key={h.chunk_id}
                    onClick={() => setActiveId(h.chunk_id)}
                    className="flex-none w-32 rounded-xl overflow-hidden border-2 border-transparent hover:border-accent transition-all"
                  >
                    <div className="aspect-video bg-slate-900 relative">
                      {h.thumbnail_url ? (
                        <img
                          src={h.thumbnail_url}
                          alt={h.camera_id}
                          className="absolute inset-0 w-full h-full object-cover"
                        />
                      ) : (
                        <div className="absolute inset-0 grid place-items-center text-slate-500 text-[10px] font-mono">
                          no thumb
                        </div>
                      )}
                      <div className="absolute bottom-1 right-1 px-1 rounded bg-black/70 text-[9px] font-mono text-white/80">
                        {sp > 0 ? `${sp}%` : s.toLocaleTimeString()}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
