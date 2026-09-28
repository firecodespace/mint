import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { getStats, listMemories } from "./api";
import type { Memory, Stats } from "./types";
import { Capture } from "./components/Capture";
import { Search } from "./components/Search";
import { Browser } from "./components/Browser";
import { ActivityLog, type ActivityEvent } from "./components/ActivityLog";

type Tab = "capture" | "search" | "memory";

let eventSeq = 0;

function App() {
  const [tab, setTab] = useState<Tab>("capture");
  const [stats, setStats] = useState<Stats | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [booting, setBooting] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const pushEvent = useCallback((kind: ActivityEvent["kind"], message: string) => {
    setEvents((prev) =>
      [{ id: ++eventSeq, time: new Date(), kind, message }, ...prev].slice(0, 100),
    );
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([getStats(), listMemories()]);
      setStats(s);
      setMemories(m);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    (async () => {
      // The first call blocks until the engine (and, on first run, the model
      // download) is ready. Retry a few times before giving up.
      for (let attempt = 0; attempt < 30; attempt++) {
        try {
          await refresh();
          pushEvent("system", "Edge memory engine online — offline & ready.");
          setBooting(false);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      setBooting(false);
    })();
  }, [refresh, pushEvent]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>Mint</h1>
            <p>Edge Memory · Field Agent Console</p>
          </div>
        </div>
        <div className="badges">
          <span className="badge offline" title="All retrieval runs on-device">
            ◍ On-device
          </span>
          <span className="badge">
            {stats ? `${stats.total}` : "—"} memories
          </span>
        </div>
      </header>

      {error && <div className="error-bar">⚠ {error}</div>}

      <nav className="tabs">
        <button className={tab === "capture" ? "on" : ""} onClick={() => setTab("capture")}>
          Capture
        </button>
        <button className={tab === "search" ? "on" : ""} onClick={() => setTab("search")}>
          Search
        </button>
        <button className={tab === "memory" ? "on" : ""} onClick={() => setTab("memory")}>
          Memory <span className="count">{memories.length}</span>
        </button>
      </nav>

      <div className="layout">
        <main className="panel">
          {booting ? (
            <div className="booting">
              <div className="spinner" />
              <p>Starting the edge memory engine…</p>
              <small>First launch downloads the embedding model once, then runs fully offline.</small>
            </div>
          ) : tab === "capture" ? (
            <Capture
              onAdded={(m) => {
                pushEvent("ingest", `Captured “${m.title || m.kind}” (${m.kind}).`);
                refresh();
              }}
              onError={(e) => pushEvent("error", e)}
            />
          ) : tab === "search" ? (
            <Search onEvent={pushEvent} />
          ) : (
            <Browser
              memories={memories}
              onDeleted={(id) => {
                pushEvent("delete", `Deleted memory ${id.slice(0, 8)}…`);
                refresh();
              }}
              onError={(e) => pushEvent("error", e)}
            />
          )}
        </main>

        <aside className="activity">
          <ActivityLog events={events} />
        </aside>
      </div>
    </div>
  );
}

export default App;
