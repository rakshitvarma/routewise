import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { routeTask, type RouteResponse } from "../api";
import { CATEGORY_META, FairwindLogo } from "../assets";
import { EXAMPLES } from "../examples";
import { CategoryPill, ModelBadge, Chip } from "./Badge";

type HistoryEntry = RouteResponse & { prompt: string; id: number };

export function Playground({
  onResult,
}: {
  onResult: (entry: HistoryEntry) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await routeTask(prompt);
      setResult(res);
      onResult({ ...res, prompt, id: Date.now() });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/60 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="mb-4 flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => setPrompt(ex.prompt)}
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-sm text-neutral-300 transition hover:border-[#3B7BFF]/50 hover:bg-neutral-800 hover:text-white"
          >
            <span className="mr-1.5">{CATEGORY_META[ex.category]?.icon}</span>
            {ex.label}
          </button>
        ))}
      </div>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Type a task, or pick an example above…"
        rows={4}
        className="w-full resize-none rounded-xl border border-neutral-800 bg-neutral-900/80 p-4 text-[15px] text-neutral-100 placeholder-neutral-600 outline-none focus:border-[#3B7BFF]/60"
      />

      <div className="mt-4 flex items-center justify-between gap-3">
        <button
          onClick={submit}
          disabled={loading || !prompt.trim()}
          className="flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#0A4DBB] to-[#3B7BFF] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[#0A4DBB]/25 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading && <FairwindLogo size={18} className="animate-fairwind-pulse" />}
          {loading ? "Routing…" : "Route & Answer →"}
        </button>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-900/50 bg-red-950/30 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-5 rounded-xl border border-neutral-800 bg-gradient-to-b from-neutral-900 to-neutral-950 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CategoryPill category={result.category} />
            <div className="flex items-center gap-2">
              <ModelBadge source={result.source} />
              <Chip>{result.tokens} tokens</Chip>
              <Chip>{result.elapsed}s</Chip>
            </div>
          </div>
          <div className="prose prose-invert prose-sm mt-4 max-w-none border-t border-neutral-800 pt-4 text-neutral-200">
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

export type { HistoryEntry };
