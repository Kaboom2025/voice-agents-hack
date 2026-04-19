import { api, type ClipHit } from "../api";

export function ResultCard({ hit }: { hit: ClipHit }) {
  const start = new Date(hit.start_ts_ms);
  return (
    <a
      href={api.clipUrl(hit.chunk_id)}
      className="block glass rounded-2xl overflow-hidden hover:shadow-glass hover:-translate-y-0.5 transition-all"
    >
      <div className="aspect-video bg-slate-100 grid place-items-center text-mute text-xs font-mono">
        {hit.thumbnail_url ? (
          <img src={hit.thumbnail_url} alt="" className="w-full h-full object-cover" />
        ) : (
          "thumbnail"
        )}
      </div>
      <div className="p-3 space-y-1">
        <div className="flex items-center justify-between text-[11px] font-mono text-mute">
          <span>{hit.camera_id}</span>
          <span>{start.toLocaleTimeString()}</span>
        </div>
        <div className="text-sm leading-snug line-clamp-2 text-ink">{hit.caption ?? "(no caption)"}</div>
        <div className="text-[10px] text-mute font-mono">score {hit.score.toFixed(2)}</div>
      </div>
    </a>
  );
}
