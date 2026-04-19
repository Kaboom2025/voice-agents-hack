import { type ClipHit } from "../api";

export function ResultCard({ hit, rank, onOpen }: { hit: ClipHit; rank?: number; onOpen?: () => void }) {
  const start = new Date(hit.start_ts_ms);
  const scorePct = Math.round(hit.score * 100);
  const scoreColor =
    scorePct >= 70 ? "bg-emerald-500" : scorePct >= 40 ? "bg-amber-500" : "bg-slate-400";

  return (
    <div
      className="group relative glass rounded-2xl overflow-hidden hover:shadow-glass hover:-translate-y-0.5 transition-all cursor-pointer"
      onClick={onOpen}
      role={onOpen ? "button" : undefined}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-slate-900 relative">
        {hit.thumbnail_url ? (
          <img
            src={hit.thumbnail_url}
            alt={`${hit.camera_id} at ${start.toLocaleTimeString()}`}
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 grid place-items-center text-slate-500 text-xs font-mono">
            no thumbnail
          </div>
        )}

        {/* Score badge */}
        <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm">
          <span className={`size-1.5 rounded-full ${scoreColor}`} />
          <span className="text-[11px] font-mono text-white/90">
            {hit.score > 0 ? `${scorePct}%` : "recent"}
          </span>
        </div>

        {/* Rank badge */}
        {rank !== undefined && (
          <div className="absolute top-2 left-2 size-6 rounded-full bg-black/70 backdrop-blur-sm grid place-items-center text-[11px] font-mono text-white/90 font-bold">
            {rank}
          </div>
        )}

        {/* Camera ID overlay */}
        <div className="absolute bottom-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm text-[11px] font-mono text-white/90">
          <span className="size-1.5 rounded-full bg-accent" />
          {hit.camera_id}
        </div>

        {/* Timestamp overlay */}
        <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm text-[10px] font-mono text-white/80">
          {start.toLocaleTimeString()}
        </div>
      </div>

      {/* Caption */}
      <div className="p-3">
        <div className="text-sm leading-snug line-clamp-2 text-ink">
          {hit.caption ?? "(no description)"}
        </div>
      </div>
    </div>
  );
}
