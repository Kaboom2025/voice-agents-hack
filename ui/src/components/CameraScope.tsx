import { useQuery } from "@tanstack/react-query";
import { api, type Camera } from "../api";

export type Scope =
  | { kind: "all" }
  | { kind: "single"; id: string }
  | { kind: "select"; ids: string[] };

export function scopeToIds(scope: Scope, all: Camera[]): string[] {
  if (scope.kind === "all") return all.map((c) => c.id);
  if (scope.kind === "single") return [scope.id];
  return scope.ids;
}

export function CameraScope({ value, onChange }: { value: Scope; onChange: (s: Scope) => void }) {
  const { data: cameras = [] } = useQuery({ queryKey: ["cameras"], queryFn: api.listCameras });

  const setMode = (kind: Scope["kind"]) => {
    if (kind === "all") onChange({ kind: "all" });
    else if (kind === "single") onChange({ kind: "single", id: cameras[0]?.id ?? "" });
    else onChange({ kind: "select", ids: cameras.map((c) => c.id) });
  };

  const toggleId = (id: string) => {
    if (value.kind !== "select") return;
    const has = value.ids.includes(id);
    onChange({ kind: "select", ids: has ? value.ids.filter((x) => x !== id) : [...value.ids, id] });
  };

  return (
    <div className="flex items-center gap-3 text-sm flex-wrap">
      <div className="inline-flex glass-soft rounded-full p-0.5">
        {(["all", "select", "single"] as const).map((k) => (
          <button
            key={k}
            onClick={() => setMode(k)}
            className={`px-3 py-1 rounded-full text-xs tracking-wide transition-colors ${
              value.kind === k ? "bg-ink text-white" : "text-mute hover:text-ink"
            }`}
          >
            {k}
          </button>
        ))}
      </div>

      {value.kind === "single" && (
        <select
          className="glass-soft rounded-full px-3 py-1 text-xs font-mono focus:outline-none"
          value={value.id}
          onChange={(e) => onChange({ kind: "single", id: e.target.value })}
        >
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>{c.id}</option>
          ))}
        </select>
      )}

      {value.kind === "select" && (
        <div className="flex flex-wrap gap-1.5">
          {cameras.map((c) => {
            const on = value.ids.includes(c.id);
            return (
              <button
                key={c.id}
                onClick={() => toggleId(c.id)}
                className={`px-2.5 py-1 rounded-full text-xs font-mono transition-colors ${
                  on
                    ? "bg-ink text-white"
                    : "glass-soft text-mute hover:text-ink"
                }`}
              >
                {c.id}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
