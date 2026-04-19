// Speech-to-text abstraction. The default provider uses the browser's
// Web Speech API so the template runs out of the box. Swap in an
// on-device Cactus/Gemma STT provider later without touching the UI.

export interface STTProvider {
  start(handlers: { onPartial?: (t: string) => void; onFinal: (t: string) => void; onError?: (e: unknown) => void }): void;
  stop(): void;
  readonly available: boolean;
}

type SpeechRecognitionCtor = new () => {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: (e: { results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void;
  onerror: (e: unknown) => void;
};

function getCtor(): SpeechRecognitionCtor | undefined {
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

class WebSpeechProvider implements STTProvider {
  private rec: ReturnType<SpeechRecognitionCtor> | null = null;
  readonly available = !!getCtor();

  start(h: Parameters<STTProvider["start"]>[0]) {
    const Ctor = getCtor();
    if (!Ctor) {
      h.onError?.(new Error("Web Speech API not available in this browser"));
      return;
    }
    const rec = new Ctor();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.onresult = (e) => {
      let partial = "";
      let final = "";
      for (let i = 0; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) final += r[0].transcript;
        else partial += r[0].transcript;
      }
      if (partial) h.onPartial?.(partial);
      if (final) h.onFinal(final);
    };
    rec.onerror = (e) => h.onError?.(e);
    rec.start();
    this.rec = rec;
  }

  stop() {
    this.rec?.stop();
    this.rec = null;
  }
}

export const stt: STTProvider = new WebSpeechProvider();
