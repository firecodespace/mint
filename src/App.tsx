// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { chatStatus, getStats, listMemories, runMaintenance } from "./api";
import type { ChatStatus, MaintenanceReport, Memory, Stats } from "./types";
import { Chat } from "./components/Chat";
import { Capture } from "./components/Capture";
import { Search } from "./components/Search";
import { Browser } from "./components/Browser";
import { MemoryGraph } from "./components/MemoryGraph";
import { Vault } from "./components/Vault";
import { Sync } from "./components/Sync";
import { Timeline } from "./components/Timeline";
import { HowItWorks } from "./components/HowItWorks";

type Tab = "chat" | "memory" | "vault" | "timeline" | "sync" | "how";
type MemView = "graph" | "list";

function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [memView, setMemView] = useState<MemView>("graph");
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [booting, setBooting] = useState(true);
  const [maintBusy, setMaintBusy] = useState(false);
  const [maintReport, setMaintReport] = useState<MaintenanceReport | null>(null);

  async function runMaint() {
    setMaintBusy(true);
    try {
      const r = await runMaintenance();
      setMaintReport(r);
      await refresh();
    } catch (e) {
      console.error(e);
    } finally {
      setMaintBusy(false);
    }
  }

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
    }, 20000);
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
        <button className={tab === "vault" ? "on" : ""} onClick={() => setTab("vault")}>
          Vault
        </button>
        <button className={tab === "timeline" ? "on" : ""} onClick={() => setTab("timeline")}>
          Timeline
        </button>
        <button className={tab === "sync" ? "on" : ""} onClick={() => setTab("sync")}>
          Sync
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
            <span className="muted">{memories.length} active memories</span>
            <div className="maint">
              {maintReport && (
                <span className="muted maint-report">
                  {maintReport.summaries} summaries · {maintReport.topics_summarized} topics
                  refreshed · {maintReport.topics_rehomed} re-filed · {maintReport.archived} archived
                </span>
              )}
              <button className="ghost" onClick={runMaint} disabled={maintBusy}>
                {maintBusy ? "Consolidating…" : "Consolidate & tidy"}
              </button>
            </div>
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
      ) : tab === "vault" ? (
        <div className="vault-tab">
          <Vault memories={memories} onChange={refresh} />
        </div>
      ) : tab === "timeline" ? (
        <Timeline />
      ) : tab === "sync" ? (
        <Sync onChange={refresh} />
      ) : (
        <HowItWorks />
      )}
    </div>
  );
}

export default App;
