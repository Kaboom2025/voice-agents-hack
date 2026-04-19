import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";
import { api, type PlaybackInfo } from "../api";

interface ReplayPlayerProps {
  cameraId: string;
  /** Wall-clock ms to start playback at (usually the retrieved chunk's start_ts_ms). */
  startTsMs: number;
  className?: string;
  /** Called once the player has successfully seeked to startTsMs. */
  onReady?: () => void;
}

/**
 * HLS replay player. Loads the camera's rolling HLS playlist and seeks to
 * `startTsMs` using the `PROGRAM-DATE-TIME` tags emitted by the follower's
 * ffmpeg sidecar. Playback continues forward naturally from there — it's
 * up to the user to stop watching.
 *
 * Safari plays HLS natively via `<video src="...m3u8">`; every other
 * browser needs hls.js to do the muxing. We branch on `Hls.isSupported()`.
 */
export function ReplayPlayer({ cameraId, startTsMs, className, onReady }: ReplayPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "missing" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setErrorMessage(null);

    const init = async () => {
      let info: PlaybackInfo;
      try {
        info = await api.playback(cameraId, startTsMs);
      } catch (err) {
        if (!cancelled) {
          setStatus("error");
          setErrorMessage(err instanceof Error ? err.message : String(err));
        }
        return;
      }
      if (cancelled) return;
      if (!info.available) {
        setStatus("missing");
        return;
      }

      const video = videoRef.current;
      if (!video) return;

      const seekToWallClock = () => {
        // hls.js exposes programDateTime on each fragment; we walk the
        // playlist to find the fragment covering `startTsMs` and seek the
        // <video>'s currentTime to (fragment.start + offset_in_fragment).
        const hls = hlsRef.current;
        if (!hls || hls.levels.length === 0) return;
        const details = hls.levels[hls.currentLevel]?.details ?? hls.levels[0]?.details;
        if (!details?.fragments?.length) return;
        const frags = details.fragments;
        let hit = frags[0];
        for (const f of frags) {
          const pdt = f.programDateTime;
          if (pdt !== null && pdt !== undefined && pdt <= startTsMs) hit = f;
          else if (pdt && pdt > startTsMs) break;
        }
        const pdt = hit.programDateTime ?? 0;
        const offsetSec = Math.max(0, (startTsMs - pdt) / 1000);
        video.currentTime = hit.start + offsetSec;
        video.play().catch(() => {});
        setStatus("ready");
        onReady?.();
      };

      if (Hls.isSupported()) {
        const hls = new Hls({
          // Keep the whole window loaded so the user can scrub backwards.
          backBufferLength: 120,
          liveSyncDurationCount: 3,
        });
        hlsRef.current = hls;
        hls.loadSource(info.playlist_url);
        hls.attachMedia(video);
        hls.on(Hls.Events.LEVEL_LOADED, seekToWallClock);
        hls.on(Hls.Events.ERROR, (_evt, data) => {
          if (data.fatal) {
            setStatus("error");
            setErrorMessage(`${data.type}: ${data.details}`);
          }
        });
      } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        // Safari / iOS native HLS path. We can't inspect PDT here, so fall
        // back to seeking via the `#t=` media fragment after metadata loads.
        video.src = info.playlist_url;
        const onMeta = () => {
          // Without PDT access, we just start from the live edge. The
          // retrieval timestamp still shows in the header for context.
          video.play().catch(() => {});
          setStatus("ready");
          onReady?.();
        };
        video.addEventListener("loadedmetadata", onMeta, { once: true });
      } else {
        setStatus("error");
        setErrorMessage("HLS playback is not supported in this browser");
      }
    };

    init();

    return () => {
      cancelled = true;
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
    // startTsMs / cameraId changes re-init the player; onReady is stable upstream.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, startTsMs]);

  return (
    <div className={className ?? "relative w-full h-full bg-slate-950"}>
      <video
        ref={videoRef}
        controls
        playsInline
        className="w-full h-full object-contain"
      />
      {status !== "ready" && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60 pointer-events-none">
          <div className="text-center text-[11px] font-mono uppercase tracking-wider text-slate-300 px-4">
            {status === "loading" && "loading recording…"}
            {status === "missing" && "no recording for this camera yet"}
            {status === "error" && (errorMessage ?? "playback error")}
          </div>
        </div>
      )}
    </div>
  );
}
