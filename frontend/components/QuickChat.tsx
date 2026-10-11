"use client";

import { AnimatePresence, motion } from "motion/react";
import { Send } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  streamChat,
  type ChatMessage,
  type Health,
} from "@/lib/api";
import { Banner, Button, Card, Select, Textarea } from "./ui";

export function QuickChat({
  health,
  provider,
  setProvider,
}: {
  health: Health | null;
  provider: string;
  setProvider: (p: string) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [live, setLive] = useState(""); // in-progress assistant text
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, live]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setInput("");
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setLive("");
    let acc = "";
    try {
      await streamChat(next, provider || null, (delta) => {
        acc += delta;
        setLive(acc);
      });
      setMessages([...next, { role: "assistant", content: acc }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLive("");
      setBusy(false);
    }
  }

  const offline = !health;

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4 px-4 py-8">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Quick chat</h2>
          <p className="text-sm text-muted">
            Raw streaming chat over the model port (SSE).
          </p>
        </div>
        <Select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          disabled={offline}
          aria-label="Provider"
          className="w-auto"
        >
          {health?.providers.map((p) => (
            <option key={p.name} value={p.name} disabled={!p.ready}>
              {p.name}
              {p.ready ? "" : " (not ready)"}
            </option>
          ))}
        </Select>
      </div>

      {offline && (
        <Banner tone="warn">
          Backend is offline — start the FastAPI server to chat.
        </Banner>
      )}

      <Card className="flex min-h-[320px] flex-col gap-3 p-4">
        {messages.length === 0 && !live && (
          <p className="m-auto text-sm text-muted">
            Say hello to test the stream ✨
          </p>
        )}
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                m.role === "user"
                  ? "self-end bg-accent text-white"
                  : "self-start border border-border bg-white/5"
              }`}
            >
              {m.content}
            </motion.div>
          ))}
        </AnimatePresence>
        {live && (
          <div className="max-w-[85%] self-start rounded-2xl border border-border bg-white/5 px-4 py-2.5 text-sm leading-relaxed">
            <span className="cursor-blink">{live}</span>
          </div>
        )}
        <div ref={endRef} />
      </Card>

      {error && <Banner tone="danger">{error}</Banner>}

      <div className="flex items-end gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Type a message…  (Enter to send, Shift+Enter for newline)"
          rows={2}
          disabled={offline}
        />
        <Button onClick={send} disabled={busy || offline} className="h-[46px]">
          <Send size={16} />
          {busy ? "…" : "Send"}
        </Button>
      </div>
    </div>
  );
}
