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

      <main className="flex-1 px-6 pb-6 space-y-6 max-w-6xl w-full mx-auto">
        <section className="space-y-3">
          <CameraScope value={scope} onChange={setScope} />
          <LiveGrid cameraIds={ids} />
        </section>

        <section className="space-y-4">
          <VoiceBar
            onSubmit={(t) => ask.mutate(t)}
            placeholder="Ask the cameras anything…"
          />

          {ask.isPending && (
            <div className="text-mute text-sm px-4">Searching the index…</div>
          )}

          {last && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 glass rounded-3xl p-6 space-y-4">
                <div className="text-[11px] uppercase tracking-wider text-mute font-mono">
                  {last.question}
                </div>
                <div className="markdown-body">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {last.answer.answer}
                  </ReactMarkdown>
                </div>
                {last.answer.citations.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-3 border-t border-edge/60">
                    {last.answer.citations.map((c) => (
                      <a
                        key={c.chunk_id}
                        href={api.clipUrl(c.chunk_id)}
                        className="px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-mono text-mute hover:text-accent hover:border-accent/40 transition-colors"
                      >
                        {c.camera_id} · {new Date(c.start_ts_ms).toLocaleTimeString()}
                      </a>
                    ))}
                  </div>
                )}
              </div>
              <div className="space-y-3">
                {last.answer.hits.map((h) => (
                  <ResultCard key={h.chunk_id} hit={h} />
                ))}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
