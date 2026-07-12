import { MODEL_INFO } from "../assets";

const CATEGORY_LABELS: Record<string, string[]> = {
  "qwen3": ["factual", "sentiment", "ner", "summarization", "code_debug", "code_gen"],
  "minimax": ["factual", "sentiment", "ner", "summarization", "math", "logic"],
  "kimi": ["code_debug", "code_gen"],
  "gemma": ["sentiment (bonus)"],
};

const LOCAL_HINTS = new Set(["qwen3"]);

export function ModelsGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {MODEL_INFO.map((m) => (
        <div
          key={m.hint}
          className="rounded-xl border border-neutral-800 bg-neutral-950/60 p-5 transition hover:border-neutral-700"
        >
          <div className="flex items-center gap-3">
            <div
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full ring-2 ring-black/30"
              style={{ background: m.bg }}
            >
              {m.icon}
            </div>
            <div>
              <div className="font-semibold text-neutral-100">{m.name}</div>
              <div
                className={
                  "text-xs font-medium " +
                  (LOCAL_HINTS.has(m.hint) ? "text-emerald-400" : "text-sky-400")
                }
              >
                {LOCAL_HINTS.has(m.hint) ? "Local · 0 tokens" : "Fireworks"}
              </div>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {(CATEGORY_LABELS[m.hint] || []).map((c) => (
              <span
                key={c}
                className="rounded-md bg-neutral-900 px-2 py-0.5 text-[11px] text-neutral-400"
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
