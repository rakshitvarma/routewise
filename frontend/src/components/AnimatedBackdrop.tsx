import { useEffect, useRef } from "react";

type Blob = {
  x: number;
  y: number;
  r: number;
  dx: number;
  dy: number;
  color: string;
};

// Lightweight canvas-based flowing gradient blobs, in the spirit of the
// Gemma model page's fluid hero background - a handful of soft radial
// gradients drifting and blending via "lighter" composite, no WebGL/3D
// library needed for this effect and it stays cheap on low-end devices.
export function AnimatedBackdrop({ className = "" }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const colors = ["#0A4DBB", "#3B7BFF", "#5CC8FC", "#1A1A1A"];
    const blobs: Blob[] = Array.from({ length: 5 }, (_, i) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: 90 + Math.random() * 110,
      dx: (Math.random() - 0.5) * 0.35,
      dy: (Math.random() - 0.5) * 0.35,
      color: colors[i % colors.length],
    }));

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.globalCompositeOperation = "lighter";
      for (const b of blobs) {
        b.x += b.dx;
        b.y += b.dy;
        if (b.x < -b.r) b.x = width + b.r;
        if (b.x > width + b.r) b.x = -b.r;
        if (b.y < -b.r) b.y = height + b.r;
        if (b.y > height + b.r) b.y = -b.r;

        const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
        grad.addColorStop(0, `${b.color}55`);
        grad.addColorStop(1, `${b.color}00`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };

    if (prefersReducedMotion) {
      // Draw a single static frame instead of animating.
      draw();
      cancelAnimationFrame(raf);
    } else {
      draw();
    }

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`pointer-events-none absolute inset-0 h-full w-full opacity-70 blur-2xl ${className}`}
      aria-hidden="true"
    />
  );
}
