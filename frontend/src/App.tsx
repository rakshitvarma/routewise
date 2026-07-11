import { useState } from "react";
import { FairwindLogo } from "./assets";
import { Playground, type HistoryEntry } from "./components/Playground";
import { ModelsGrid } from "./components/ModelsGrid";
import { History } from "./components/History";
import { Stats } from "./components/Stats";
import { AnimatedBackdrop } from "./components/AnimatedBackdrop";
import { FeatureCards } from "./components/FeatureCards";

const GITHUB_URL = "https://github.com/rakshitvarma/fairwind";
const DOCKER_URL = "https://github.com/rakshitvarma/fairwind/pkgs/container/fairwind";

function App() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const totalTokens = history.reduce((sum, h) => sum + h.tokens, 0);

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      <nav className="sticky top-0 z-20 border-b border-neutral-900/80 bg-black/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3.5">
          <div className="flex items-center gap-2.5">
            <FairwindLogo size={28} />
            <span className="text-lg font-bold tracking-tight text-white">Fairwind</span>
          </div>
          <div className="flex items-center gap-5 text-sm text-neutral-400">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="transition hover:text-white">
              GitHub
            </a>
            <a href={DOCKER_URL} target="_blank" rel="noreferrer" className="transition hover:text-white">
              Docker image
            </a>
          </div>
        </div>
      </nav>

      <div className="relative overflow-hidden">
        <main className="relative mx-auto max-w-5xl px-6 pb-24 pt-14">
          <div className="hero-card mb-10">
            <div className="hero-card-spin" />
            <div className="hero-card-content px-6 py-14 text-center sm:px-12">
              <AnimatedBackdrop className="h-full" />
              <div className="relative">
                <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
                  Route every task to the{" "}
                  <span className="bg-gradient-to-r from-[#3B7BFF] via-[#5CC8FC] to-[#3B7BFF] bg-clip-text text-transparent">
                    cheapest model
                  </span>{" "}
                  that can answer it
                </h1>
                <p className="mx-auto mt-4 max-w-2xl text-[15px] leading-relaxed text-neutral-400">
                  A hybrid routing agent that classifies each task for free, answers it locally when a
                  small bundled model can be trusted, and only spends Fireworks tokens on what
                  genuinely needs it. Built for AMD Developer Hackathon Act II, Track 1.
                </p>
              </div>
            </div>
          </div>

          <div className="mb-8">
            <Stats tokens={totalTokens} queries={history.length} />
          </div>

          <Playground onResult={(e) => setHistory((h) => [e, ...h])} />

          <History entries={history} />
        </main>
      </div>

      <main className="mx-auto max-w-5xl px-6 pb-24">
        <section className="mt-4">
          <h2 className="mb-1 text-xl font-bold text-white">Models in play</h2>
          <p className="mb-5 text-sm text-neutral-500">
            Local models answer for free when they can be trusted; everything else routes to the
            cheapest sufficient Fireworks model.
          </p>
          <ModelsGrid />
        </section>

        <section className="mt-16">
          <h2 className="mb-1 text-xl font-bold text-white">How it works</h2>
          <p className="mb-5 text-sm text-neutral-500">
            Every task moves down this pipeline until something can answer it confidently.
          </p>
          <FeatureCards />
        </section>
      </main>

      <footer className="border-t border-neutral-900 py-8 text-center text-sm text-neutral-600">
        Built for AMD Developer Hackathon Act II · Track 1
      </footer>
    </div>
  );
}

export default App;
