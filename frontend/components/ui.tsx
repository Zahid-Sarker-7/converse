"use client";

import { motion, type HTMLMotionProps } from "motion/react";
import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// ---- Button ----------------------------------------------------------------

type ButtonVariant = "primary" | "ghost" | "subtle" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none select-none";

const BTN_VARIANT: Record<ButtonVariant, string> = {
  primary:
    "text-white bg-accent hover:brightness-110 shadow-[0_8px_30px_-8px_rgba(139,124,255,0.7)]",
  ghost: "text-text bg-white/5 hover:bg-white/10 border border-border",
  subtle: "text-muted hover:text-text hover:bg-white/5",
  danger: "text-white bg-danger hover:brightness-110",
};

const BTN_SIZE: Record<ButtonSize, string> = {
  sm: "text-sm px-3 py-1.5",
  md: "text-sm px-4 py-2.5",
  lg: "text-base px-6 py-3",
};

export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  ...props
}: {
  variant?: ButtonVariant;
  size?: ButtonSize;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <motion.button
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.97 }}
      className={cn(BTN_BASE, BTN_VARIANT[variant], BTN_SIZE[size], className)}
      {...(props as HTMLMotionProps<"button">)}
    >
      {children}
    </motion.button>
  );
}

// ---- Card ------------------------------------------------------------------

export function Card({
  className,
  children,
  ...props
}: { className?: string; children: ReactNode } & HTMLMotionProps<"div">) {
  return (
    <motion.div
      className={cn("glass rounded-2xl", className)}
      {...props}
    >
      {children}
    </motion.div>
  );
}

// ---- Status pill -----------------------------------------------------------

export function StatusPill({
  tone,
  children,
}: {
  tone: "ok" | "down" | "warn";
  children: ReactNode;
}) {
  const color =
    tone === "ok"
      ? "text-ok"
      : tone === "warn"
        ? "text-warn"
        : "text-danger";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border border-border bg-surface/70 px-3 py-1 text-xs font-medium backdrop-blur",
        color,
      )}
    >
      <motion.span
        className="inline-block h-2 w-2 rounded-full bg-current"
        animate={tone === "ok" ? { opacity: [1, 0.35, 1] } : {}}
        transition={{ duration: 2, repeat: Infinity }}
      />
      {children}
    </span>
  );
}

// ---- Chip ------------------------------------------------------------------

export function Chip({
  children,
  tone = "default",
  className,
}: {
  children: ReactNode;
  tone?: "default" | "accent" | "warn";
  className?: string;
}) {
  const tones = {
    default: "border-border bg-white/5 text-muted",
    accent: "border-accent/40 bg-accent/10 text-accent",
    warn: "border-warn/40 bg-warn/10 text-warn",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

// ---- Form fields -----------------------------------------------------------

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-muted/70">{hint}</span>}
    </label>
  );
}

const INPUT_CLASS =
  "w-full rounded-lg border border-border bg-bg/50 px-3 py-2 text-sm text-text placeholder:text-muted/60 focus:border-accent focus:outline-none transition-colors";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn(INPUT_CLASS, props.className)} />;
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(INPUT_CLASS, "resize-y leading-relaxed", props.className)}
    />
  );
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(INPUT_CLASS, "cursor-pointer", props.className)}
    />
  );
}

// ---- Skeleton --------------------------------------------------------------

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("shimmer rounded-lg", className)} />;
}

// ---- Inline banner (errors / notices) --------------------------------------

export function Banner({
  tone = "danger",
  children,
}: {
  tone?: "danger" | "warn" | "info";
  children: ReactNode;
}) {
  const tones = {
    danger: "border-danger/40 bg-danger/10 text-danger",
    warn: "border-warn/40 bg-warn/10 text-warn",
    info: "border-accent/40 bg-accent/10 text-accent",
  };
  return (
    <div
      role="status"
      className={cn(
        "rounded-xl border px-4 py-3 text-sm leading-relaxed",
        tones[tone],
      )}
    >
      {children}
    </div>
  );
}
