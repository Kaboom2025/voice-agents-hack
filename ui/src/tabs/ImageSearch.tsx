import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, type ClipHit, type QueryResponse } from "../api";
import { CameraScope, scopeToIds, type Scope } from "../components/CameraScope";

/**
 * Image Vector Search — a debug/testing page that runs the text query through
 * the leader's embedding pipeline and displays the raw video-modality hits
 * (thumbnails + cosine scores) with no LLM synthesis. Useful for eyeballing
 * how well the image embeddings align with specific text queries.
 */
export function ImageSearch() {
  const [scope, setScope] = useState<Scope>({ kind: "all" });
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(24);
  const [last, setLast] = useState<{ question: string; answer: QueryResponse } | null>(null);

  const { data: cameras = [] } = useQuery({ queryKey: ["cameras"], queryFn: api.listCameras });
  const ids = scopeToIds(scope, cameras);

  const search = useMutation({
    mutationFn: async (question: string) => {
      const res = await api.query({
        query: question,
        cameras: scope.kind === "all" ? undefined : ids,
        top_k: topK,
        modalities: ["video"],
      });
      setLast({ question, answer: res });
      return res;
    },
  });

  const hits = last?.answer.hits ?? [];
  // Sort by score descending so the most-similar frames surface at the top
  // (the backend already sorts but we re-sort defensively in case RRF merging
  // produced a different ordering).
  const sorted = [...hits].sort((a, b) => b.score - a.score);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    search.mutate(q);
  };

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <CameraScope value={scope} onChange={setScope} />
      </section>

      <section className="space-y-4">
        <form onSubmit={onSubmit} className="glass rounded-3xl p-4 flex items-center gap-3">
          <span className="text-[11px] uppercase tracking-wider text-mute font-mono pl-2">
            image search
          </span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. a person wearing a red shirt"
            className="flex-1 bg-transparent outline-none text-ink placeholder:text-mute text-sm py-2"
            autoFocus
          />
          <label className="flex items-center gap-2 text-[11px] font-mono text-mute">
            top_k
            <input
              type="number"
              min={1}
              max={200}
              value={topK}
              onChange={(e) => setTopK(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
              className="w-16 bg-slate-100 border border-slate-200 rounded-lg px-2 py-1 text-ink text-right"
            />
          </label>
          <button
            type="submit"
            disabled={search.isPending || !query.trim()}
            className="px-4 py-2 rounded-full bg-ink text-white text-sm font-medium disabled:opacity-40"
          >
            {search.isPending ? "Searching…" : "Search"}
          </button>
        </form>

        {search.isError && (
          <div className="glass rounded-2xl p-4 text-sm text-red-600">
            {(search.error as Error).message}
          </div>
        )}
      </section>

      {last && (
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="text-[11px] uppercase tracking-wider text-mute font-mono">
              Image hits for: "{last.question}"
            </div>
            <div className="text-[11px] font-mono text-mute">
              — {sorted.length} result{sorted.length !== 1 ? "s" : ""}
            </div>
          </div>

          {sorted.length === 0 ? (
            <div className="glass rounded-3xl p-12 text-center text-mute text-sm">
              No matching frames. Try a different phrasing.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sorted.map((h, i) => (
                <ImageHitCard key={h.chunk_id} hit={h} rank={i + 1} />
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function ImageHitCard({ hit, rank }: { hit: ClipHit; rank: number }) {
  const start = new Date(hit.start_ts_ms);
  const scorePct = Math.max(0, Math.min(100, Math.round(hit.score * 100)));
  const barColor =
    scorePct >= 70 ? "bg-emerald-500" : scorePct >= 40 ? "bg-amber-500" : "bg-slate-400";

  return (
    <div className="glass rounded-2xl overflow-hidden">
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

        <div className="absolute top-2 left-2 size-6 rounded-full bg-black/70 backdrop-blur-sm grid place-items-center text-[11px] font-mono text-white/90 font-bold">
          {rank}
        </div>

        <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm text-[11px] font-mono text-white/90">
          {hit.score.toFixed(4)}
        </div>

        <div className="absolute bottom-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm text-[11px] font-mono text-white/90">
          <span className="size-1.5 rounded-full bg-accent" />
          {hit.camera_id}
        </div>

        <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded-full bg-black/70 backdrop-blur-sm text-[10px] font-mono text-white/80">
          {start.toLocaleTimeString()}
        </div>
      </div>

      <div className="p-3 space-y-2">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-slate-200 overflow-hidden">
            <div className={`h-full ${barColor}`} style={{ width: `${scorePct}%` }} />
          </div>
          <div className="text-[11px] font-mono text-mute w-10 text-right">{scorePct}%</div>
        </div>
        <div className="text-xs leading-snug line-clamp-2 text-mute">
          {hit.caption ?? "(no caption)"}
        </div>
        <div className="text-[10px] font-mono text-mute truncate" title={hit.chunk_id}>
          {hit.chunk_id}
        </div>
      </div>
    </div>
  );
}
