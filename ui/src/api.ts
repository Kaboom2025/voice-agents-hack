// Typed contract for the leader's HTTP/WS surface.
// Mirrors PRD §5.4 (POST /query) plus the Watch and Command modes.

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

export type SearchResponse = {
  hits: ClipHit[];
  took_ms: number;
};

export type AnswerRequest = {
  query: string;
  chunk_ids: string[];
};

export type AnswerResponse = {
  answer: string;
  citations: Citation[];
  took_ms: number;
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

export type PlaybackInfo = {
  camera_id: string;
  playlist_url: string;
  start_ts_ms?: number;
  available: boolean;
};

export interface ApiClient {
  listCameras(): Promise<Camera[]>;
  query(req: QueryRequest): Promise<QueryResponse>;
  search(req: QueryRequest): Promise<SearchResponse>;
  answer(req: AnswerRequest): Promise<AnswerResponse>;
  rpcStream(req: RpcRequest): AsyncIterable<RpcEvent>;
  liveStream(cameraIds: string[], signal: AbortSignal): AsyncIterable<LiveFrame>;
  clipUrl(chunkId: string): string;
  playback(cameraId: string, startTsMs?: number): Promise<PlaybackInfo>;
}

// ────────────────────────── HTTP client ──────────────────────────
// Talks to the leader's axum server (proxied through Vite at /api).

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

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

  async search(req) {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`search: ${res.status} ${await res.text()}`);
    return (await res.json()) as SearchResponse;
  },

  async answer(req) {
    const res = await fetch("/api/answer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`answer: ${res.status} ${await res.text()}`);
    return (await res.json()) as AnswerResponse;
  },

  async *rpcStream(req) {
    const res = await fetch("/api/rpc", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`rpcStream: ${res.status} ${await res.text()}`);
    const data = (await res.json()) as RpcEvent[];
    for (const ev of data) {
      yield ev;
    }
  },

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

  clipUrl(chunkId) {
    return `/api/clips/${encodeURIComponent(chunkId)}`;
  },

  async playback(cameraId, startTsMs) {
    const qs = new URLSearchParams({ camera_id: cameraId });
    if (startTsMs !== undefined) qs.set("start_ts_ms", String(startTsMs));
    const res = await fetch(`/api/playback?${qs.toString()}`);
    if (!res.ok) throw new Error(`playback: ${res.status}`);
    return (await res.json()) as PlaybackInfo;
  },
};

export const api: ApiClient = httpClient;
