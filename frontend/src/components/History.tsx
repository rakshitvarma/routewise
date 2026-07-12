import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import type { HistoryEntry } from "./Playground";
import { CategoryPill, ModelBadge, Chip } from "./Badge";

export function History({ entries }: { entries: HistoryEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="mt-8">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Session history
      </h3>
      <div className="space-y-2">
        {entries.map((e) => (
          <HistoryRow key={e.id} entry={e} />
        ))}
      </div>
    </div>
  );
}

function HistoryRow({ entry }: { entry: HistoryEntry }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/40">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left"
      >
        <div className="flex min-w-0 items-center gap-3">
          <CategoryPill category={entry.category} />
          <span className="truncate text-sm text-neutral-500">
            {entry.prompt.slice(0, 70)}
            {entry.prompt.length > 70 ? "…" : ""}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <ModelBadge source={entry.source} />
          <Chip>{entry.tokens} tokens</Chip>
        </div>
      </button>
      {open && (
        <div className="prose prose-invert prose-sm max-w-none border-t border-neutral-800 px-4 py-3 text-neutral-300">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
          >
            {entry.answer}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}
