import { useRef } from "react";

type Feature = {
  icon: string;
  title: string;
  body: string;
};

const FEATURES: Feature[] = [
  {
    icon: "🧭",
    title: "Local classifier",
    body: "Regex/keyword heuristics route every task into one of 8 categories - zero tokens, zero model calls.",
  },
  {
    icon: "🧮",
    title: "Deterministic math",
    body: "Bare arithmetic gets solved instantly and for free. Word problems fall through to Fireworks.",
  },
  {
    icon: "⚡",
    title: "Two bundled local models",
    body: "Factual, sentiment, NER, summarisation, and code answer entirely for free - sanity- and syntax-checked before being trusted.",
  },
  {
    icon: "🧩",
    title: "Self-consistent logic",
    body: "Constraint puzzles get 3-call majority vote, run concurrently - cheap insurance against a demonstrated flaky-reasoning failure mode.",
  },
  {
    icon: "🔗",
    title: "Merged Fireworks batches",
    body: "Everything local can't resolve is grouped by model and answered in as few calls as possible, not one call per category.",
  },
];

function FeatureCard({ feature }: { feature: Feature }) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--x", `${e.clientX - rect.left}px`);
    el.style.setProperty("--y", `${e.clientY - rect.top}px`);
  };

  return (
    <div
      ref={ref}
      onMouseMove={handleMove}
      className="spotlight-card group relative overflow-hidden rounded-2xl border border-neutral-800 bg-gradient-to-b from-neutral-900 to-neutral-950 p-6 transition-transform duration-300 hover:-translate-y-1 hover:border-neutral-700"
    >
      <div className="relative z-10">
        <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-[#0A4DBB]/15 text-xl ring-1 ring-[#0A4DBB]/30 transition group-hover:scale-110 group-hover:bg-[#0A4DBB]/25">
          {feature.icon}
        </div>
        <h3 className="mb-2 text-base font-semibold text-white">{feature.title}</h3>
        <p className="text-sm leading-relaxed text-neutral-400">{feature.body}</p>
      </div>
    </div>
  );
}

export function FeatureCards() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {FEATURES.map((f) => (
        <FeatureCard key={f.title} feature={f} />
      ))}
    </div>
  );
}
