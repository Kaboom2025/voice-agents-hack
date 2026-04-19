// Typed contract for the leader's HTTP/WS surface.
// Mirrors PRD §5.4 (POST /query) plus the Watch and Command modes.
// The mock client returns fixtures so the UI is buildable before the
// axum server inside `leader` exists. Swap `mockClient` for a real
// `httpClient` later — nothing else has to change.

export type CameraStatus = "online" | "offline" | "degraded";

export type Camera = {
  id: string;
  follower_node_id: string;
  status: CameraStatus;
  last_seen_ms: number;
  chunks_per_min: number;
};

export type Citation = {
  camera_id: string;
  chunk_id: string;
  start_ts_ms: number;
  end_ts_ms: number;
};

export type ClipHit = {
  chunk_id: string;
  camera_id: string;
  start_ts_ms: number;
  end_ts_ms: number;
  caption?: string;
  score: number;
  thumbnail_url?: string;
};

export type Modality = "video" | "audio" | "caption";

export type QueryRequest = {
  query: string;
  cameras?: string[];
  time_range?: { from_ms: number; to_ms: number };
  top_k?: number;
  modalities?: Modality[];
};

export type QueryResponse = {
  answer: string;
  citations: Citation[];
  hits: ClipHit[];
};

export type RpcMethod = "GetConfig" | "SetConfig" | "Ping" | "RotateKeys";

export type RpcRequest = {
  cameras: string[];
  method: RpcMethod;
  params?: Record<string, unknown>;
};

export type RpcEvent = {
  camera_id: string;
  chunk: string;
  done: boolean;
};

export type LiveFrame = {
  camera_id: string;
  ts_ms: number;
  // In production this is an encoded frame (JPEG/VP8). The mock yields a
  // poster URL so the template renders without a real video transport.
  poster_url: string;
};

export interface ApiClient {
  listCameras(): Promise<Camera[]>;
  query(req: QueryRequest): Promise<QueryResponse>;
  rpcStream(req: RpcRequest): AsyncIterable<RpcEvent>;
  liveStream(cameraIds: string[], signal: AbortSignal): AsyncIterable<LiveFrame>;
  clipUrl(chunkId: string): string;
}

// ────────────────────────── mock implementation ──────────────────────────

const FIXTURE_CAMERAS: Camera[] = [
  { id: "cam-front-door", follower_node_id: "node-1", status: "online", last_seen_ms: Date.now(), chunks_per_min: 12 },
  { id: "cam-lab-1", follower_node_id: "node-2", status: "online", last_seen_ms: Date.now(), chunks_per_min: 12 },
  { id: "cam-lab-2", follower_node_id: "node-3", status: "degraded", last_seen_ms: Date.now() - 30_000, chunks_per_min: 4 },
  { id: "cam-loading-bay", follower_node_id: "node-4", status: "offline", last_seen_ms: Date.now() - 600_000, chunks_per_min: 0 },
];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const mockClient: ApiClient = {
  async listCameras() {
    await sleep(120);
    return FIXTURE_CAMERAS;
  },

  async query(req) {
    await sleep(400);
    const cams = req.cameras ?? FIXTURE_CAMERAS.map((c) => c.id);
    const now = Date.now();
    const hits: ClipHit[] = cams.slice(0, 3).map((cam, i) => ({
      chunk_id: `${cam}-${1000 + i}`,
      camera_id: cam,
      start_ts_ms: now - (i + 1) * 60_000,
      end_ts_ms: now - (i + 1) * 60_000 + 5_000,
      caption: `Synthetic match #${i + 1} on ${cam}`,
      score: 0.9 - i * 0.07,
    }));
    return {
      answer: `Mock synthesis for "${req.query}". Replace mockClient with the real HTTP client to get LLM-generated answers grounded in the multicam index.`,
      citations: hits.map((h) => ({
        camera_id: h.camera_id,
        chunk_id: h.chunk_id,
        start_ts_ms: h.start_ts_ms,
        end_ts_ms: h.end_ts_ms,
      })),
      hits,
    };
  },

  async *rpcStream(req) {
    for (const camera_id of req.cameras) {
      await sleep(150);
      yield { camera_id, chunk: `[${req.method}] ack from ${camera_id}\n`, done: false };
      await sleep(150);
      yield { camera_id, chunk: `ok\n`, done: true };
    }
  },

  async *liveStream(cameraIds, signal) {
    while (!signal.aborted) {
      for (const camera_id of cameraIds) {
        if (signal.aborted) return;
        yield {
          camera_id,
          ts_ms: Date.now(),
          poster_url: "",
        };
      }
      await sleep(1000);
    }
  },

  clipUrl(chunkId) {
    return `/api/clips/${encodeURIComponent(chunkId)}`;
  },
};

// ────────────────────────── real HTTP client ──────────────────────────
// Talks to the leader's axum server (proxied through Vite at /api).
// `liveStream` and `listCameras` are real; query/rpcStream/clipUrl still
// fall through to the mock until those leader endpoints exist.

export function liveJpegUrl(cameraId: string, cacheBust?: number): string {
  const t = cacheBust ?? performance.now();
  return `/api/live/${encodeURIComponent(cameraId)}?t=${t}`;
}

export const httpClient: ApiClient = {
  async listCameras() {
    const res = await fetch("/api/cameras");
    if (!res.ok) throw new Error(`listCameras: ${res.status}`);
    return (await res.json()) as Camera[];
  },

  async query(req) {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`query: ${res.status} ${await res.text()}`);
    return (await res.json()) as QueryResponse;
  },
  rpcStream: mockClient.rpcStream,

  // The real live transport is browser-driven `<img>` polling against
  // `liveJpegUrl(cameraId)`. This generator only emits LiveFrame ticks for
  // any consumer that wants poster URLs; LiveTile bypasses it.
  async *liveStream(cameraIds, signal) {
    while (!signal.aborted) {
      for (const camera_id of cameraIds) {
        if (signal.aborted) return;
        const ts_ms = Date.now();
        yield { camera_id, ts_ms, poster_url: liveJpegUrl(camera_id, ts_ms) };
      }
      await sleep(33);
    }
  },

  clipUrl: mockClient.clipUrl,
};

export const api: ApiClient = httpClient;
