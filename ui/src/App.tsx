import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { api, type QueryResponse } from "./api";
import { CameraScope, scopeToIds, type Scope } from "./components/CameraScope";
import { LiveGrid } from "./components/LiveGrid";
import { VoiceBar } from "./components/VoiceBar";
import { ResultCard } from "./components/ResultCard";

function hitsGridCols(n: number): string {
  if (n <= 1) return "grid-cols-1";
  if (n <= 2) return "grid-cols-2";
  if (n <= 3) return "grid-cols-3";
  return "grid-cols-4";
}

export default function App() {
  const [scope, setScope] = useState<Scope>({ kind: "all" });
  const [last, setLast] = useState<{ question: string; answer: QueryResponse } | null>(null);

  const { data: cameras = [] } = useQuery({ queryKey: ["cameras"], queryFn: api.listCameras });
  const ids = scopeToIds(scope, cameras);

  const ask = useMutation({
    mutationFn: async (question: string) => {
      const res = await api.query({
        query: question,
        cameras: scope.kind === "all" ? undefined : ids,
        top_k: 12,
      });
      setLast({ question, answer: res });
      return res;
    },
  });

  const hits = last?.answer.hits ?? [];
  const hasHits = hits.length > 0;

  return (
    <div className="min-h-full flex flex-col">
      <header className="flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2.5">
          <span className="size-2 rounded-full bg-accent" />
          <span className="font-mono text-xs tracking-[0.2em] text-mute uppercase">multicam</span>
        </div>
        <div className="text-[11px] font-mono text-mute">
          {ids.length} stream{ids.length === 1 ? "" : "s"}
        </div>
      </header>

      <main className="flex-1 px-6 pb-6 space-y-6 max-w-7xl w-full mx-auto">
        {/* Live camera feeds */}
        <section className="space-y-3">
          <CameraScope value={scope} onChange={setScope} />
          <LiveGrid cameraIds={ids} />
        </section>

        {/* Query bar */}
        <section className="space-y-4">
          <VoiceBar
            onSubmit={(t) => ask.mutate(t)}
            placeholder="Ask the cameras anything…"
          />

          {ask.isPending && (
            <div className="flex items-center gap-2 text-mute text-sm px-1">
              <span className="size-2 rounded-full bg-accent animate-pulse" />
              Searching across {ids.length} camera{ids.length !== 1 ? "s" : ""}…
            </div>
          )}
        </section>

        {/* === Results: clips first, answer below === */}
        {last && (
          <section className="space-y-6">
            {/* Query echo */}
            <div className="flex items-center gap-3">
              <div className="text-[11px] uppercase tracking-wider text-mute font-mono">
                Results for: "{last.question}"
              </div>
              <div className="text-[11px] font-mono text-mute">
                — {hits.length} clip{hits.length !== 1 ? "s" : ""} found
              </div>
            </div>

            {/* Clip grid — full-width, prominent */}
            {hasHits ? (
              <div className={`grid ${hitsGridCols(hits.length)} gap-4`}>
                {hits.map((h, i) => (
                  <ResultCard key={h.chunk_id} hit={h} rank={i + 1} />
                ))}
              </div>
            ) : (
              <div className="glass rounded-3xl p-12 text-center text-mute text-sm">
                No matching clips found. Try broadening your query or time range.
              </div>
            )}

            {/* LLM answer + citations below the clips */}
            <div className="glass rounded-3xl p-6 space-y-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-mute font-mono">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="size-4" aria-hidden>
                  <path d="M12 2a4 4 0 0 1 4 4v1h1a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-8a3 3 0 0 1 3-3h1V6a4 4 0 0 1 4-4Z" strokeLinecap="round" strokeLinejoin="round" />
                  <circle cx="9" cy="14" r="1" fill="currentColor" />
                  <circle cx="15" cy="14" r="1" fill="currentColor" />
                </svg>
                AI Analysis
              </div>
              <div className="markdown-body text-ink">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {last.answer.answer}
                </ReactMarkdown>
              </div>

              {/* Source citations */}
              {last.answer.citations.length > 0 && (
                <div className="space-y-2 pt-3 border-t border-edge/60">
                  <div className="text-[11px] uppercase tracking-wider text-mute font-mono">
                    Sources
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {last.answer.citations.map((c) => (
                      <span
                        key={c.chunk_id}
                        className="px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-mono text-mute"
                      >
                        {c.camera_id} · {new Date(c.start_ts_ms).toLocaleTimeString()}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
