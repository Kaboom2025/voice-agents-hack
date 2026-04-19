import { useState } from "react";
import { stt } from "../voice";

function MicIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}

function StopIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <rect x="7" y="7" width="10" height="10" rx="2" />
    </svg>
  );
}

export function VoiceBar({
  placeholder = "Ask a question, or describe what you want to see…",
  onSubmit,
}: {
  placeholder?: string;
  onSubmit: (text: string) => void;
}) {
  const [text, setText] = useState("");
  const [listening, setListening] = useState(false);

  const submit = () => {
    const t = text.trim();
    if (!t) return;
    onSubmit(t);
    setText("");
  };

  const toggleMic = () => {
    if (listening) {
      stt.stop();
      setListening(false);
      return;
    }
    if (!stt.available) {
      alert("Web Speech API not available in this browser. Plug in a Cactus STT provider in src/voice.ts.");
      return;
    }
    setListening(true);
    stt.start({
      onPartial: (t) => setText(t),
      onFinal: (t) => {
        setText("");
        setListening(false);
        onSubmit(t);
      },
      onError: () => setListening(false),
    });
  };

  return (
    <div className="glass rounded-full flex items-center gap-2 px-2 py-1.5 w-full">
      <button
        onClick={toggleMic}
        className={`relative shrink-0 size-9 rounded-full grid place-items-center transition-colors ${
          listening
            ? "bg-accent text-white"
            : "bg-slate-100 text-ink hover:text-accent"
        }`}
        aria-label={listening ? "Stop listening" : "Start voice input"}
        title={listening ? "Listening — click to stop" : "Voice input"}
      >
        {listening && (
          <span className="absolute inset-0 rounded-full bg-accent/50 animate-ping" />
        )}
        <span className="relative">
          {listening ? <StopIcon className="size-4" /> : <MicIcon className="size-4" />}
        </span>
      </button>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder={placeholder}
        className="flex-1 bg-transparent px-2 py-1 text-sm placeholder:text-mute focus:outline-none"
      />
      <button
        onClick={submit}
        className="shrink-0 px-4 py-1.5 rounded-full bg-ink text-white text-sm font-medium hover:opacity-90"
      >
        Send
      </button>
    </div>
  );
}
