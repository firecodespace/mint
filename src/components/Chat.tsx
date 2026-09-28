import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  chat,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
} from "../api";
import type {
  CapturedItem,
  ChatMessage,
  ChatStatus,
  ConversationSummary,
  RetrievedItem,
  StageEvent,
  TokenEvent,
} from "../types";

interface Turn {
  role: "user" | "assistant";
  content: string;
}

interface FlowEntry {
  id: number;
  stage: string;
  detail: string;
}

const STAGE_LABEL: Record<string, string> = {
  retrieving: "Retrieving memory",
  retrieved: "Memory retrieved",
  thinking: "Thinking",
  answering: "Answering",
  extracting: "Extracting memories",
  done: "Turn complete",
};

let flowSeq = 0;

export function Chat({
  status,
  onCaptured,
}: {
  status: ChatStatus | null;
  onCaptured: () => void;
}) {
  const [convos, setConvos] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState<ConversationSummary | null>(null);

  // Live per-turn cognition state.
  const [thinking, setThinking] = useState("");
  const [answer, setAnswer] = useState("");
  const [flow, setFlow] = useState<FlowEntry[]>([]);
  const [retrieved, setRetrieved] = useState<RetrievedItem[]>([]);
  const [captured, setCaptured] = useState<CapturedItem[]>([]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const onCapturedRef = useRef(onCaptured);
  onCapturedRef.current = onCaptured;

  async function loadConvos() {
    try {
      setConvos(await listConversations());
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadConvos();
  }, []);

  // Wire Tauri streaming events once. The `disposed` guard makes the async
  // listen() setup safe under React StrictMode's mount/unmount/remount, so we
  // never end up with duplicate listeners (which doubled every token).
  useEffect(() => {
    let disposed = false;
    const uns: Array<() => void> = [];
    const track = (u: () => void) => {
      if (disposed) u();
      else uns.push(u);
    };
    (async () => {
      track(
        await listen<StageEvent>("chat:stage", (e) => {
          const { stage, detail } = e.payload;
          setFlow((prev) => [...prev, { id: ++flowSeq, stage, detail }].slice(-40));
        }),
      );
      track(
        await listen<TokenEvent>("chat:token", (e) => {
          const { channel, text } = e.payload;
          if (channel === "thinking") setThinking((t) => t + text);
          else setAnswer((a) => a + text);
        }),
      );
      track(
        await listen<CapturedItem>("chat:captured", (e) => {
          setCaptured((prev) => [...prev, e.payload]);
          onCapturedRef.current();
        }),
      );
    })();
    return () => {
      disposed = true;
      uns.forEach((u) => u());
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, answer, busy]);

  function newChat() {
    setActiveId(null);
    setTurns([]);
    setThinking("");
    setAnswer("");
    setFlow([]);
    setRetrieved([]);
    setCaptured([]);
    setInput("");
  }

  async function openChat(id: string) {
    try {
      const c = await getConversation(id);
      if (!c) return;
      setActiveId(c.id);
      setTurns(c.messages.map((m) => ({ role: m.role, content: m.content })));
      setThinking("");
      setAnswer("");
      setFlow([]);
      setRetrieved([]);
      setCaptured([]);
    } catch (e) {
      console.error(e);
    }
  }

  async function confirmDelete(deleteMemories: boolean) {
    if (!deleting) return;
    const id = deleting.id;
    setDeleting(null);
    try {
      await deleteConversation(id, deleteMemories);
      if (activeId === id) newChat();
      await loadConvos();
      if (deleteMemories) onCapturedRef.current();
    } catch (e) {
      console.error(e);
    }
  }

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    let convoId = activeId;
    if (!convoId) {
      try {
        const c = await createConversation();
        convoId = c.id;
        setActiveId(c.id);
      } catch (err) {
        console.error(err);
        return;
      }
    }

    const history: ChatMessage[] = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setBusy(true);
    setThinking("");
    setAnswer("");
    setFlow([]);
    setRetrieved([]);
    setCaptured([]);

    try {
      const result = await chat(convoId, text, history);
      setTurns((prev) => [...prev, { role: "assistant", content: result.answer }]);
      setRetrieved(result.retrieved);
      setCaptured(result.captured);
      setAnswer("");
      loadConvos();
    } catch (err) {
      setTurns((prev) => [...prev, { role: "assistant", content: `Error: ${String(err)}` }]);
    } finally {
      setBusy(false);
    }
  }

  const offline = status && !status.ollama_up;

  return (
    <div className="chat-view">
      <aside className="chat-sidebar">
        <button className="new-chat" onClick={newChat}>
          + New chat
        </button>
        <div className="convo-list">
          {convos.length === 0 && <p className="muted side-empty">No conversations yet.</p>}
          {convos.map((c) => (
            <div
              key={c.id}
              className={`convo ${activeId === c.id ? "on" : ""}`}
              onClick={() => openChat(c.id)}
            >
              <div className="convo-main">
                <span className="convo-title">{c.title || "Untitled"}</span>
                <span className="convo-meta">
                  {c.message_count} msgs · {c.memory_count} mem
                </span>
              </div>
              <button
                className="convo-del"
                title="Delete conversation"
                onClick={(ev) => {
                  ev.stopPropagation();
                  setDeleting(c);
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="chat-main">
        <div className="messages" ref={scrollRef}>
          {turns.length === 0 && !busy && (
            <div className="empty-chat">
              <h2>Talk to your local memory</h2>
              <p>
                Answered on-device and distilled into searchable memory. Nothing leaves
                this machine.
              </p>
              {offline && (
                <p className="warn-line">
                  Ollama is not reachable on localhost:11434. Start Ollama to chat.
                </p>
              )}
            </div>
          )}

          {turns.map((t, i) => (
            <div key={i} className={`msg ${t.role}`}>
              <div className="msg-role">{t.role === "user" ? "You" : "Mint"}</div>
              <div className="msg-body">{t.content}</div>
            </div>
          ))}

          {busy && (
            <div className="msg assistant">
              <div className="msg-role">Mint</div>
              <div className="msg-body">
                {answer || <span className="typing">thinking…</span>}
              </div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={send}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(e);
              }
            }}
            placeholder="Message Mint…  (Enter to send, Shift+Enter for newline)"
            rows={2}
            disabled={busy}
          />
          <button className="primary" type="submit" disabled={busy || !input.trim()}>
            {busy ? "…" : "Send"}
          </button>
        </form>
      </section>

      <aside className="cognition">
        <div className="cog-section">
          <h3>Cognition</h3>
          {thinking ? (
            <pre className="thinking">{thinking}</pre>
          ) : (
            <p className="muted">The model's live reasoning appears here.</p>
          )}
        </div>
        <div className="cog-section">
          <h3>Retrieved memory</h3>
          {retrieved.length === 0 ? (
            <p className="muted">Grounding memories for the turn show here.</p>
          ) : (
            <ul className="cog-list">
              {retrieved.map((r) => (
                <li key={r.id}>
                  <span className={`pill ${r.kind}`}>{r.kind}</span>
                  <span className="cog-title">{r.title || "(untitled)"}</span>
                  <span className="cog-score">{r.score.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="cog-section">
          <h3>Captured this turn</h3>
          {captured.length === 0 ? (
            <p className="muted">New memories distilled from the turn show here.</p>
          ) : (
            <ul className="cog-list">
              {captured.map((c) => (
                <li key={c.id}>
                  <span className={`pill ${c.kind}`}>{c.kind}</span>
                  <span className="cog-title">{c.title}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <footer className="planflow">
        <span className="pf-label">Plan flow</span>
        <div className="pf-track">
          {flow.length === 0 && (
            <span className="muted">Pipeline stages stream here as the turn runs.</span>
          )}
          {flow.map((f) => (
            <span key={f.id} className={`pf-step ${f.stage}`}>
              {STAGE_LABEL[f.stage] || f.stage}
              {f.detail ? <em> {f.detail}</em> : null}
            </span>
          ))}
        </div>
      </footer>

      {deleting && (
        <div className="modal-backdrop" onClick={() => setDeleting(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Delete conversation</h3>
            <p>
              Delete <strong>{deleting.title || "Untitled"}</strong>? It has{" "}
              {deleting.memory_count} linked{" "}
              {deleting.memory_count === 1 ? "memory" : "memories"}.
            </p>
            <div className="modal-actions">
              <button className="ghost" onClick={() => setDeleting(null)}>
                Cancel
              </button>
              <button className="ghost" onClick={() => confirmDelete(false)}>
                Delete chat only
              </button>
              <button
                className="ghost danger"
                onClick={() => confirmDelete(true)}
                disabled={deleting.memory_count === 0}
              >
                Delete chat + memories
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
