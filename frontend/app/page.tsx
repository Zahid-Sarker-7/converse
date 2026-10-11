"use client";

import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { MessageSquare, Sparkles, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";
import { getHealth, type Health } from "@/lib/api";
import { Hero } from "@/components/Hero";
import { QuickChat } from "@/components/QuickChat";
import { RoleplayStudio } from "@/components/RoleplayStudio";
import { StatusPill, cn } from "@/components/ui";

type View = "hero" | "studio" | "chat";

const TABS: { id: View; label: string; icon: typeof Sparkles }[] = [
  { id: "hero", label: "Home", icon: Wand2 },
  { id: "studio", label: "Role-play", icon: Sparkles },
  { id: "chat", label: "Quick chat", icon: MessageSquare },
];

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [provider, setProvider] = useState("");
  const [view, setView] = useState<View>("hero");

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setProvider(h.default_provider);
      })
      .catch(() => setHealth(null));
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-bg/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
          <button
            onClick={() => setView("hero")}
            className="text-lg font-bold tracking-tight"
          >
            Con<span className="text-gradient">verse</span>
          </button>

          <nav className="hidden items-center gap-1 rounded-full border border-border bg-surface/60 p-1 sm:flex">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setView(t.id)}
                className={cn(
                  "relative rounded-full px-4 py-1.5 text-sm transition-colors",
                  view === t.id ? "text-text" : "text-muted hover:text-text",
                )}
              >
                {view === t.id && (
                  <motion.span
                    layoutId="tab-pill"
                    className="absolute inset-0 rounded-full bg-white/10"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                  />
                )}
                <span className="relative flex items-center gap-1.5">
                  <t.icon size={14} />
                  {t.label}
                </span>
              </button>
            ))}
          </nav>

          {health ? (
            <StatusPill tone="ok">{health.app} v{health.version}</StatusPill>
          ) : (
            <StatusPill tone="down">offline</StatusPill>
          )}
        </div>

        {/* mobile tabs */}
        <nav className="flex items-center gap-1 overflow-x-auto px-4 pb-2 sm:hidden">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              className={cn(
                "flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-sm",
                view === t.id
                  ? "bg-white/10 text-text"
                  : "text-muted",
              )}
            >
              <t.icon size={14} />
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-h-[calc(100dvh-57px)]">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
          >
            {view === "hero" && (
              <Hero
                health={health}
                onStart={() => setView("studio")}
                onQuickChat={() => setView("chat")}
              />
            )}
            {view === "studio" && (
              <RoleplayStudio health={health} provider={provider} />
            )}
            {view === "chat" && (
              <QuickChat
                health={health}
                provider={provider}
                setProvider={setProvider}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </MotionConfig>
  );
}
