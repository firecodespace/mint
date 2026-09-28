import { useState } from "react";
import { searchMemories } from "../api";
import type { MemoryKind, SearchMode, SearchResponse } from "../types";
import type { ActivityEvent } from "./ActivityLog";

const MODES: SearchMode[] = ["hybrid", "dense", "sparse"];
const KIND_FILTERS: (MemoryKind | "")[] = [
  "",
  "observation",
  "note",
  "measurement",
  "event",
  "doc_chunk",
];

export function Search({
  onEvent,
}: {
  onEvent: (kind: ActivityEvent["kind"], message: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [site, setSite] = useState("");
  const [kind, setKind] = useState<MemoryKind | "">("");
  const [resp, setResp] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    try {
      const r = await searchMemories({
        query: query.trim(),
        mode,
        limit: 10,
        site_id: site.trim() || null,
        kind: kind || null,
      });
      setResp(r);
      onEvent(
        "search",
        `“${query.trim()}” · ${mode} · ${r.results.length} hits in ${r.latency_ms.toFixed(1)} ms`,
      );
    } catch (err) {
      onEvent("error", String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="search">
      <h2>Search device memory</h2>
      <form onSubmit={run}>
        <div className="searchbar">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask the device… e.g. pump pressure problems at site A"
            autoFocus
          />
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "…" : "Search"}
          </button>
        </div>

        <div className="controls">
          <div className="seg">
            {MODES.map((m) => (
              <button
                type="button"
                key={m}
                className={mode === m ? "on" : ""}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            ))}
          </div>
          <input
            className="filter"
            value={site}
            onChange={(e) => setSite(e.target.value)}
            placeholder="site filter"
          />
          <select value={kind} onChange={(e) => setKind(e.target.value as MemoryKind | "")}>
            {KIND_FILTERS.map((k) => (
              <option key={k || "any"} value={k}>
                {k || "any kind"}
              </option>
            ))}
          </select>
        </div>
      </form>

      {resp && (
        <div className="results">
          <div className="results-head">
            <span>{resp.results.length} results</span>
            <span className="latency" title="On-device retrieval latency">
              ⚡ {resp.latency_ms.toFixed(1)} ms · {resp.mode}
            </span>
          </div>
          {resp.results.length === 0 && <p className="muted">No matches on device.</p>}
          <ul>
            {resp.results.map((r) => (
              <li key={r.memory.id} className="result">
                <div className="result-top">
                  <span className={`pill ${r.memory.kind}`}>{r.memory.kind}</span>
                  <strong>{r.memory.title || "(untitled)"}</strong>
                  <span className="score">{r.score.toFixed(3)}</span>
                </div>
                <p className="result-text">{r.memory.text}</p>
                <div className="result-meta">
                  {r.memory.site_id && <span>site: {r.memory.site_id}</span>}
                  {r.memory.asset_id && <span>asset: {r.memory.asset_id}</span>}
                  {r.memory.tags.map((t) => (
                    <span key={t} className="tag">
                      #{t}
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
