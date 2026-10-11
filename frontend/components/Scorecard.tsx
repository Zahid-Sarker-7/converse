"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { ChevronDown, Quote, Target } from "lucide-react";
import { useState } from "react";
import type { CriterionScore, Scorecard as ScorecardData } from "@/lib/api";
import { ScoreGauge } from "./ScoreGauge";
import { Card } from "./ui";

function scoreColor(score: number): string {
  if (score <= 2) return "var(--color-danger)";
  if (score === 3) return "var(--color-warn)";
  return "var(--color-ok)";
}

function CriterionRow({
  c,
  index,
  reduce,
}: {
  c: CriterionScore;
  index: number;
  reduce: boolean | null;
}) {
  const [open, setOpen] = useState(false);
  const pct = (c.score / 5) * 100;
  const color = scoreColor(c.score);

  return (
    <motion.div
      variants={{
        hidden: { opacity: 0, y: 14 },
        show: { opacity: 1, y: 0 },
      }}
      className="rounded-xl border border-border bg-white/[0.02]"
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left"
        aria-expanded={open}
      >
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="truncate text-sm font-medium capitalize">
              {c.criterion}
            </span>
            <span
              className="shrink-0 text-sm font-bold tabular-nums"
              style={{ color }}
            >
              {c.score}
              <span className="text-muted">/5</span>
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-border/60">
            <motion.div
              className="h-full rounded-full"
              style={{ background: color }}
              initial={{ width: reduce ? `${pct}%` : 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.9, delay: 0.15 + index * 0.08, ease: "easeOut" }}
            />
          </div>
        </div>
        <ChevronDown
          size={18}
          className={`shrink-0 text-muted transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="space-y-3 px-4 pb-4 pt-1">
              <blockquote className="flex gap-2 rounded-lg border-l-2 border-accent/60 bg-accent/5 px-3 py-2 text-sm italic text-text/90">
                <Quote size={14} className="mt-0.5 shrink-0 text-accent" />
                <span>{c.evidence}</span>
              </blockquote>
              <p className="text-sm leading-relaxed text-muted">{c.rationale}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function Scorecard({ card }: { card: ScorecardData }) {
  const reduce = useReducedMotion();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-5"
    >
      {/* Overall + summary */}
      <Card className="flex flex-col items-center gap-5 p-6 sm:flex-row sm:items-center sm:gap-8">
        <ScoreGauge value={card.overall} />
        <div className="flex-1 text-center sm:text-left">
          <h3 className="mb-1 text-sm font-medium uppercase tracking-wide text-muted">
            Overall
          </h3>
          <p className="text-lg leading-relaxed">{card.summary}</p>
        </div>
      </Card>

      {/* Criteria — staggered */}
      <motion.div
        variants={{ show: { transition: { staggerChildren: 0.08 } } }}
        initial="hidden"
        animate="show"
        className="space-y-2.5"
      >
        {card.criteria.map((c, i) => (
          <CriterionRow key={c.criterion + i} c={c} index={i} reduce={reduce} />
        ))}
      </motion.div>

      {/* Focus next — highlighted */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 + card.criteria.length * 0.08 }}
      >
        <Card className="flex items-start gap-3 border-accent/40 bg-gradient-to-br from-accent/15 to-accent-2/5 p-5">
          <div className="rounded-lg bg-accent/20 p-2">
            <Target size={20} className="text-accent" />
          </div>
          <div>
            <h4 className="mb-1 text-sm font-semibold text-accent">
              Focus on next
            </h4>
            <p className="leading-relaxed">{card.focus_next}</p>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
}
