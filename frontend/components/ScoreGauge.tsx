"use client";

import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "motion/react";
import { useEffect, useState } from "react";

/** Radial gauge with an animated arc sweep + count-up number (value out of `max`). */
export function ScoreGauge({
  value,
  max = 5,
  size = 176,
  decimals = 1,
}: {
  value: number;
  max?: number;
  size?: number;
  decimals?: number;
}) {
  const reduce = useReducedMotion();
  const stroke = 14;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const target = Math.max(0, Math.min(1, value / max));

  const progress = useMotionValue(reduce ? target : 0);
  const dashoffset = useTransform(
    progress,
    (p) => circumference * (1 - p),
  );
  const [display, setDisplay] = useState(reduce ? value : 0);

  useEffect(() => {
    if (reduce) {
      progress.set(target);
      setDisplay(value);
      return;
    }
    const arc = animate(progress, target, { duration: 1.2, ease: "easeOut" });
    const num = animate(0, value, {
      duration: 1.2,
      ease: "easeOut",
      onUpdate: (v) => setDisplay(v),
    });
    return () => {
      arc.stop();
      num.stop();
    };
  }, [target, value, reduce, progress]);

  return (
    <div
      className="relative"
      style={{ width: size, height: size }}
      role="img"
      aria-label={`Overall score ${value} out of ${max}`}
    >
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--color-accent)" />
            <stop offset="100%" stopColor="var(--color-accent-2)" />
          </linearGradient>
        </defs>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth={stroke}
          opacity={0.5}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="url(#gauge-grad)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          style={{ strokeDashoffset: dashoffset }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tabular-nums text-gradient">
          {display.toFixed(decimals)}
        </span>
        <span className="text-xs text-muted">out of {max}</span>
      </div>
    </div>
  );
}
