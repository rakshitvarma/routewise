// Real model logos via Simple Icons (CC0), same source as the original
// Streamlit demo's webapp/assets.py. Gemma/Kimi have no dedicated icons -
// Gemini's spark (same Google DeepMind lineage) and Moonshot AI's mark
// (Kimi's maker) stand in, both clearly labelled rather than passed off
// as the thing's own logo.

// Brand mark: folded-ribbon "F" + accent dot, per the Fairwind brand sheet
// (dark-mode primary: white mark, #0A4DBB dot).
export const FairwindLogo = ({ size = 36, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" className={className}>
    <path
      d="M14 6 H34 Q37 6 35 8.6 L21 26 Q19.4 28 16.8 28 H12 Q9 28 9 25 V11 Q9 7.5 12 6.6 Z"
      fill="#FFFFFF"
    />
    <path
      d="M12 20 H26 Q29 20 27 22.6 L17.5 34.4 Q15.8 36.5 13.1 36.5 H10 Q7 36.5 7 33.5 V25 Q7 21.5 10 20.4 Z"
      fill="#FFFFFF"
      opacity="0.92"
    />
    <circle cx="32" cy="34" r="6" fill="#0A4DBB" />
  </svg>
);

const MINIMAX_PATH =
  "M11.43 3.92a.86.86 0 1 0-1.718 0v14.236a1.999 1.999 0 0 1-3.997 0V9.022a.86.86 " +
  "0 1 0-1.718 0v3.87a1.999 1.999 0 0 1-3.997 0V11.49a.57.57 0 0 1 1.139 0v1.404a.86.86 " +
  "0 0 0 1.719 0V9.022a1.999 1.999 0 0 1 3.997 0v9.134a.86.86 0 0 0 1.719 0V3.92a1.998 1.998 " +
  "0 1 1 3.996 0v11.788a.57.57 0 1 1-1.139 0zm10.572 3.105a2 2 0 0 0-1.999 1.997v7.63a.86.86 " +
  "0 0 1-1.718 0V3.923a1.999 1.999 0 0 0-3.997 0v16.16a.86.86 0 0 1-1.719 0V18.08a.57.57 0 " +
  "1 0-1.138 0v2a1.998 1.998 0 0 0 3.996 0V3.92a.86.86 0 0 1 1.719 0v12.73a1.999 1.999 0 " +
  "0 0 3.996 0V9.023a.86.86 0 1 1 1.72 0v6.686a.57.57 0 0 0 1.138 0V9.022a2 2 0 0 0-1.998-1.997";

const MOONSHOT_PATH =
  "m1.053 16.91 9.538 2.55a21 20.981 0 0 0 .06 2.031l5.956 1.592a12 11.99 0 0 1-15.554-6.172" +
  "m-1.02-5.79 11.352 3.035a21 20.981 0 0 0-.469 2.01l10.817 2.89a12 11.99 0 0 1-1.845 " +
  "2.004L.658 15.918a12 11.99 0 0 1-.625-4.796m1.593-5.146L13.573 9.17a21 20.981 0 0 " +
  "0-1.01 1.874l11.297 3.02a21 20.981 0 0 1-.67 2.362l-11.55-3.087L.125 10.26a12 11.99 " +
  "0 0 1 1.499-4.285ZM6.067 1.58l11.285 3.016a21 20.981 0 0 0-1.688 1.719l7.824 2.091a21 " +
  "20.981 0 0 1 .513 2.664L2.107 5.218a12 11.99 0 0 1 3.96-3.638M21.68 4.866 7.222 1.003A12 " +
  "11.99 0 0 1 21.68 4.866";

const GEMINI_PATH =
  "M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 " +
  "0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 " +
  "4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 " +
  "3.81 2.55t2.55 3.81";

const QWEN_PATH =
  "M23.919 14.545 20.817 9.17l1.47-2.544a.56.56 0 0 0 0-.566l-1.633-2.83a.57.57 0 0 0-.49-.283" +
  "h-6.207L12.487.402a.57.57 0 0 0-.49-.284H8.732a.56.56 0 0 0-.49.284L5.139 5.775h-2.94a.56.56 " +
  "0 0 0-.49.284L.077 8.887a.56.56 0 0 0 0 .567L3.18 14.83l-1.47 2.545a.56.56 0 0 0 0 .566l1.634 " +
  "2.83a.57.57 0 0 0 .49.283h6.205l1.47 2.545a.57.57 0 0 0 .49.284h3.266a.57.57 0 0 0 " +
  ".49-.284l3.104-5.375h2.94a.57.57 0 0 0 .49-.283l1.634-2.828a.55.55 0 0 0-.004-.568M8.733.686" +
  "l1.634 2.828-1.634 2.828H21.8L20.164 9.17H7.425L5.63 6.06Zm1.306 19.801-6.205-.002 1.634-2.83" +
  "h3.265L2.201 6.344h3.267q3.182 5.517 6.367 11.032zm10.124-5.66L18.53 12l-6.532 11.315-1.634-2.83" +
  "c2.129-3.673 4.25-7.351 6.373-11.028h3.592l3.102 5.374z";

const MiniIcon = ({ path, fill, size = 16 }: { path: string; fill: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path fill={fill} d={path} />
  </svg>
);

export type ModelInfo = {
  hint: string;
  name: string;
  icon: React.ReactNode;
  bg: string;
};

export const MODEL_INFO: ModelInfo[] = [
  { hint: "minimax", name: "MiniMax M3", icon: <MiniIcon path={MINIMAX_PATH} fill="#E73562" />, bg: "#FFFFFF" },
  { hint: "kimi", name: "Kimi (Moonshot AI)", icon: <MiniIcon path={MOONSHOT_PATH} fill="#111111" />, bg: "#FFFFFF" },
  { hint: "gemma", name: "Gemma (Google)", icon: <MiniIcon path={GEMINI_PATH} fill="#8E75B2" />, bg: "#FFFFFF" },
  { hint: "qwen-coder", name: "Qwen2.5-Coder-1.5B", icon: <MiniIcon path={QWEN_PATH} fill="#6950EF" />, bg: "#FFFFFF" },
  { hint: "qwen", name: "Qwen2.5-1.5B", icon: <MiniIcon path={QWEN_PATH} fill="#6950EF" />, bg: "#FFFFFF" },
];

const BOLT_PATH = "M13 2 3 14h7l-1 8 10-12h-7z";

export function modelBadge(source: string): { name: string; icon: React.ReactNode; bg: string } {
  const low = (source || "").toLowerCase();
  for (const m of MODEL_INFO) {
    if (low.includes(m.hint)) return m;
  }
  if (low.includes("deterministic")) {
    return { name: "Local (0 tokens)", icon: <MiniIcon path={BOLT_PATH} fill="#0E1117" size={13} />, bg: "linear-gradient(135deg,#43E97B,#38F9D7)" };
  }
  const short = source?.split("/").pop() || "n/a";
  return { name: short, icon: <span className="text-xs">?</span>, bg: "linear-gradient(135deg,#5A5F73,#3A3E4D)" };
}

export const CATEGORY_META: Record<string, { icon: string; color: string }> = {
  math: { icon: "🧮", color: "#7C5CFC" },
  factual: { icon: "📖", color: "#5CC8FC" },
  sentiment: { icon: "💬", color: "#FC5C8D" },
  summarization: { icon: "📝", color: "#5CFCA8" },
  ner: { icon: "🏷️", color: "#FCC85C" },
  code_debug: { icon: "🐛", color: "#FC8D5C" },
  code_gen: { icon: "⚙️", color: "#8D5CFC" },
  logic: { icon: "🧩", color: "#5CFCE0" },
};
