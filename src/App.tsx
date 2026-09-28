import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { chatStatus, getStats, listMemories } from "./api";
import type { ChatStatus, Memory, Stats } from "./types";
import { Chat } from "./components/Chat";
import { Capture } from "./components/Capture";
import { Search } from "./components/Search";
import { Browser } from "./components/Browser";
import { MemoryGraph } from "./components/MemoryGraph";
import { HowItWorks } from "./components/HowItWorks";

type Tab = "chat" | "memory" | "how";
type MemView = "graph" | "list";

function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [memView, setMemView] = useState<MemView>("graph");
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [booting, setBooting] = useState(true);

  const refresh = useCallback(async () => {
    const [s, m] = await Promise.all([getStats(), listMemories()]);
    setStats(s);
    setMemories(m);
  }, []);

  useEffect(() => {
    (async () => {
      for (let attempt = 0; attempt < 40; attempt++) {
        try {
          await refresh();
          setStatus(await chatStatus());
          setBooting(false);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
      setBooting(false);
    })();
  }, [refresh]);

  useEffect(() => {
    const t = setInterval(() => {
      chatStatus().then(setStatus).catch(() => {});
    }, 8000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>Mint</h1>
            <p>Edge Memory and Intelligence, running locally</p>
          </div>
        </div>
        <div className="badges">
          <span className="badge offline">On-device</span>
          <span className={`badge ${status?.ollama_up ? "ok" : "warn"}`}>
            {status?.ollama_up ? status.chat_model : "Ollama offline"}
          </span>
          <span className="badge">{stats ? `${stats.total} memories` : "—"}</span>
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "chat" ? "on" : ""} onClick={() => setTab("chat")}>
          Chat
        </button>
        <button className={tab === "memory" ? "on" : ""} onClick={() => setTab("memory")}>
          Memory <span className="count">{memories.length}</span>
        </button>
        <button className={tab === "how" ? "on" : ""} onClick={() => setTab("how")}>
          How it works
        </button>
      </nav>

      {booting ? (
        <div className="booting">
          <div className="spinner" />
          <p>Starting the edge memory engine…</p>
          <small>First launch downloads the embedding model once, then runs fully offline.</small>
        </div>
      ) : tab === "chat" ? (
        <Chat status={status} onCaptured={refresh} />
      ) : tab === "memory" ? (
        <div className="memory-tab">
          <div className="memory-toolbar">
            <div className="seg">
              <button
                className={memView === "graph" ? "on" : ""}
                onClick={() => setMemView("graph")}
              >
                Graph
              </button>
              <button
                className={memView === "list" ? "on" : ""}
                onClick={() => setMemView("list")}
              >
                List
              </button>
            </div>
            <span className="muted">{memories.length} memories on this device</span>
          </div>
          {memView === "graph" ? (
            <MemoryGraph memories={memories} />
          ) : (
            <div className="memory-view">
              <div className="memory-col">
                <Search />
                <Capture onAdded={refresh} onError={(e) => console.error(e)} />
              </div>
              <div className="memory-col">
                <Browser memories={memories} onDeleted={refresh} onError={(e) => console.error(e)} />
              </div>
            </div>
          )}
        </div>
      ) : (
        <HowItWorks />
      )}
    </div>
  );
}

export default App;
