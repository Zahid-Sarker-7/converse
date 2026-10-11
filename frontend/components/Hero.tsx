"use client";

import { motion } from "motion/react";
import { ArrowRight, MessageSquare, Mic, Sparkles } from "lucide-react";
import type { Health } from "@/lib/api";
import { Button, StatusPill } from "./ui";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};
const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

export function Hero({
  health,
  onStart,
  onQuickChat,
}: {
  health: Health | null;
  onStart: () => void;
  onQuickChat: () => void;
}) {
  return (
    <motion.section
      variants={container}
      initial="hidden"
      animate="show"
      className="mx-auto flex max-w-3xl flex-col items-center px-4 py-16 text-center sm:py-24"
    >
      <motion.div variants={item}>
        {health ? (
          <StatusPill tone="ok">
            backend live · {health.default_provider}
          </StatusPill>
        ) : (
          <StatusPill tone="down">backend offline</StatusPill>
        )}
      </motion.div>

      <motion.h1
        variants={item}
        className="mt-6 text-4xl font-bold leading-tight tracking-tight sm:text-6xl"
      >
        Rehearse the <span className="text-gradient">hard conversation</span>
        <br className="hidden sm:block" /> before it happens.
      </motion.h1>

      <motion.p
        variants={item}
        className="mt-5 max-w-xl text-lg leading-relaxed text-muted"
      >
        Spar with an in-character AI — an interviewer, a tough customer, an
        examiner — then get scored on the exact moments that mattered, with
        quoted evidence and one thing to fix next.
      </motion.p>

      <motion.div
        variants={item}
        className="mt-9 flex flex-col items-center gap-3 sm:flex-row"
      >
        <Button size="lg" onClick={onStart}>
          <Sparkles size={18} />
          Start a role-play
          <ArrowRight size={18} />
        </Button>
        <Button size="lg" variant="ghost" onClick={onQuickChat}>
          <MessageSquare size={18} />
          Quick chat
        </Button>
      </motion.div>

      <motion.div
        variants={item}
        className="mt-14 grid w-full gap-3 sm:grid-cols-3"
      >
        {[
          {
            icon: MessageSquare,
            title: "In-character",
            body: "Multi-turn role-play that stays in persona and pushes back.",
          },
          {
            icon: Sparkles,
            title: "Scored with evidence",
            body: "A rubric scorecard with a verbatim quote per criterion.",
          },
          {
            icon: Mic,
            title: "Provider-swappable",
            body: "Groq or Gemini behind one port — switch by config.",
          },
        ].map((f) => (
          <div
            key={f.title}
            className="glass rounded-2xl p-5 text-left transition-transform hover:-translate-y-1"
          >
            <f.icon size={20} className="mb-3 text-accent" />
            <h3 className="mb-1 text-sm font-semibold">{f.title}</h3>
            <p className="text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        ))}
      </motion.div>
    </motion.section>
  );
}
