import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type RpcMethod } from "../api";
import { CameraScope, scopeToIds, type Scope } from "../components/CameraScope";

const METHODS: RpcMethod[] = ["Ping", "GetConfig", "SetConfig", "RotateKeys"];

export function Command({ scope, onScopeChange }: { scope: Scope; onScopeChange: (s: Scope) => void }) {
  const { data: cameras = [] } = useQuery({ queryKey: ["cameras"], queryFn: api.listCameras });
  const [method, setMethod] = useState<RpcMethod>("Ping");
  const [params, setParams] = useState("{}");
  const [log, setLog] = useState<string[]>([]);
  const [running, setRunning] = useState(false);

  const send = async () => {
    const ids = scopeToIds(scope, cameras);
    if (ids.length === 0) return;
    let parsed: Record<string, unknown> = {};
    try {
      parsed = params.trim() ? JSON.parse(params) : {};
    } catch {
      setLog((l) => [...l, "! invalid JSON params"]);
      return;
    }
    setRunning(true);
    setLog([`> ${method} → ${ids.join(", ")}`]);
    try {
      for await (const ev of api.rpcStream({ cameras: ids, method, params: parsed })) {
        setLog((l) => [...l, `[${ev.camera_id}] ${ev.chunk.trimEnd()}`]);
      }
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <CameraScope value={scope} onChange={onScopeChange} />
        <div className="text-xs text-mute font-mono">RPC fan-out target</div>
      </div>

      <div className="flex gap-2">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as RpcMethod)}
          className="bg-panel2 border border-edge rounded px-2 py-2 text-sm"
        >
          {METHODS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <input
          value={params}
          onChange={(e) => setParams(e.target.value)}
          placeholder='params (JSON, e.g. {"key":"value"})'
          className="flex-1 bg-panel2 border border-edge rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
        />
        <button
          onClick={send}
          disabled={running}
          className="px-4 rounded bg-accent text-bg text-sm font-medium disabled:opacity-50"
        >
          {running ? "…" : "Send"}
        </button>
      </div>

      <pre className="rounded-lg border border-edge bg-panel p-3 text-xs font-mono whitespace-pre-wrap min-h-[12rem] max-h-[24rem] overflow-auto">
        {log.length === 0 ? <span className="text-mute">RPC responses stream here.</span> : log.join("\n")}
      </pre>
    </div>
  );
}
