import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { deleteMemory, listScheduled, setTaskDone } from "../api";
import type { Memory } from "../types";

type Bucket = "Overdue" | "Today" | "This week" | "Later" | "Done";
const ORDER: Bucket[] = ["Overdue", "Today", "This week", "Later", "Done"];

function startOfToday(): number {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function bucketOf(m: Memory): Bucket {
  if (m.done) return "Done";
  const today = startOfToday();
  const due = new Date((m.due_at || "") + "T00:00:00").getTime();
  if (!Number.isFinite(due)) return "Later";
  const day = 86400000;
  if (due < today) return "Overdue";
  if (due < today + day) return "Today";
  if (due < today + 7 * day) return "This week";
  return "Later";
}

function fmtDue(due: string | null): string {
  if (!due) return "";
  const d = new Date(due + "T00:00:00");
  if (!Number.isFinite(d.getTime())) return due;
  const today = startOfToday();
  const diff = Math.round((d.getTime() - today) / 86400000);
  const rel =
    diff === 0 ? "today" : diff === 1 ? "tomorrow" : diff === -1 ? "yesterday" :
    diff > 1 ? `in ${diff} days` : `${-diff} days ago`;
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} · ${rel}`;
}

export function Timeline() {
  const [items, setItems] = useState<Memory[]>([]);

  const refresh = useCallback(async () => {
    try {
      setItems(await listScheduled());
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refresh();
    let un: (() => void) | null = null;
    let disposed = false;
    (async () => {
      const u = await listen("chat:scheduled", () => refresh());
      if (disposed) u();
      else un = u;
    })();
    return () => {
      disposed = true;
      un?.();
    };
  }, [refresh]);

  async function toggle(m: Memory) {
    await setTaskDone(m.id, !m.done);
    refresh();
  }
  async function remove(id: string) {
    await deleteMemory(id);
    refresh();
  }

  const groups: Record<Bucket, Memory[]> = {
    Overdue: [], Today: [], "This week": [], Later: [], Done: [],
  };
  for (const m of items) groups[bucketOf(m)].push(m);

  return (
    <div className="timeline-tab">
      <div className="timeline-inner">
        <h1>Timeline</h1>
        <p className="how-lead">
          Deadlines, tasks, and dated events that Mint detected in your conversations.
          Mention something like "submit the application by next Friday" in chat and it
          lands here automatically.
        </p>

        {items.length === 0 && (
          <p className="muted">
            Nothing scheduled yet. Mention a date or deadline in chat and it will appear
            here.
          </p>
        )}

        {ORDER.map((b) =>
          groups[b].length === 0 ? null : (
            <section key={b} className="tl-group">
              <h3 className={`tl-head ${b === "Overdue" ? "overdue" : ""}`}>{b}</h3>
              {groups[b].map((m) => (
                <div key={m.id} className={`tl-item ${m.done ? "done" : ""}`}>
                  <button
                    className={`check ${m.done ? "on" : ""}`}
                    onClick={() => toggle(m)}
                    title={m.done ? "Mark not done" : "Mark done"}
                    aria-label={m.done ? "Mark not done" : "Mark done"}
                  />
                  <div className="tl-main">
                    <span className="tl-title">{m.title}</span>
                    <span className="tl-meta">
                      {m.tags.map((t) => (
                        <span key={t} className="tag">
                          {t}
                        </span>
                      ))}
                    </span>
                  </div>
                  <span className="tl-due">{fmtDue(m.due_at)}</span>
                  <button className="ghost danger tl-del" onClick={() => remove(m.id)}>
                    Remove
                  </button>
                </div>
              ))}
            </section>
          ),
        )}
      </div>
    </div>
  );
}
