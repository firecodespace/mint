import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  policySummary,
  setApiKey,
  setAutoSync,
  setOnline,
  setPullAll,
  setServerUrl,
  setSyncOverride,
  syncNow,
  syncStatus,
} from "../api";
import type { PolicySummary, SyncReport, SyncStatus } from "../types";

function fmtTime(epochSecs: string | null): string {
  if (!epochSecs) return "never";
  const n = Number(epochSecs);
  if (!Number.isFinite(n)) return epochSecs;
  return new Date(n * 1000).toLocaleTimeString();
}

/** Human labels for the sync policy's "stays on device" categories. */
const CATEGORY_LABEL: Record<string, string> = {
  secret: "Secrets and credentials",
  financial: "Financial details",
  government_id: "Government IDs",
  health: "Personal health",
  contact: "Contact details",
  user: "Kept local by you",
  derived: "Derived links and private topics",
};

function reportLine(r: SyncReport): string {
  const parts = [`pushed ${r.pushed}`, `pulled ${r.pulled}`];
  if (r.conflicts) parts.push(`${r.conflicts} conflict${r.conflicts > 1 ? "s" : ""} kept as versions`);
  if (r.withheld) parts.push(`${r.withheld} kept on device`);
  if (r.retracted) parts.push(`${r.retracted} retracted from cloud`);
  if (r.cloud_only) parts.push(`${r.cloud_only} left in cloud`);
  return parts.join(" · ");
}

export function Sync({ onChange }: { onChange: () => void }) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [policy, setPolicy] = useState<PolicySummary | null>(null);
  const [urlDraft, setUrlDraft] = useState("");
  const [keyDraft, setKeyDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<SyncReport | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await syncStatus();
      setStatus(s);
      setUrlDraft((prev) => (prev === "" ? s.server_url : prev));
    } catch (e) {
      console.error(e);
    }
  }, []);

  const refreshPolicy = useCallback(async () => {
    try {
      setPolicy(await policySummary());
    } catch (e) {
      console.error(e);
    }
  }, []);

  // Held in a ref so a non-memoized parent callback doesn't re-subscribe.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    refresh();
    refreshPolicy();
    const t = setInterval(refresh, 5000);
    // The background auto-sync loop announces each run.
    const un = listen("sync:done", () => {
      refresh();
      refreshPolicy();
      onChangeRef.current();
    });
    return () => {
      clearInterval(t);
      un.then((f) => f());
    };
  }, [refresh, refreshPolicy]);

  async function toggleOnline() {
    if (!status) return;
    await setOnline(!status.online);
    await refresh();
  }

  async function toggleAuto() {
    if (!status) return;
    await setAutoSync(!status.auto);
    await refresh();
  }

  async function togglePullAll() {
    if (!status) return;
    await setPullAll(!status.pull_all);
    await refresh();
  }

  async function saveUrl() {
    await setServerUrl(urlDraft.trim());
    await refresh();
  }

  async function saveKey(clear: boolean) {
    await setApiKey(clear ? "" : keyDraft.trim());
    setKeyDraft("");
    await refresh();
  }

  async function allowSync(id: string) {
    try {
      await setSyncOverride(id, true);
      await refreshPolicy();
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function runSync() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const r = await syncNow();
      setReport(r);
      await refresh();
      await refreshPolicy();
      onChange();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const online = status?.online ?? false;
  const reachable = status?.reachable ?? false;
  const c = status?.counts;
  const cats = Object.entries(policy?.local_by_category ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="sync-tab">
      <div className="sync-inner">
        <h1>Edge &lt;-&gt; Cloud sync</h1>
        <p className="how-lead">
          Everything runs offline on this device. A sync policy decides what may leave it:
          secrets, financial and ID numbers, personal health facts, and contact details stay
          here. Consolidated knowledge and documents sync; other devices' raw notes stay in the
          cloud and are searched there when you are online.
        </p>

        <div className="sync-grid">
          <div className="sync-card">
            <h3>Connectivity</h3>
            <div className="conn-row">
              <button className={`toggle ${online ? "on" : ""}`} onClick={toggleOnline}>
                <span className="knob" />
              </button>
              <div>
                <strong>{online ? "Online" : "Offline (airplane mode)"}</strong>
                <p className="muted">
                  {online
                    ? reachable
                      ? "Server reachable"
                      : "Server not reachable"
                    : "Toggle on to allow syncing"}
                </p>
              </div>
              <span className={`dot-ind ${online && reachable ? "green" : online ? "amber" : "grey"}`} />
            </div>
            <div className="conn-row">
              <button className={`toggle ${status?.auto ? "on" : ""}`} onClick={toggleAuto}>
                <span className="knob" />
              </button>
              <div>
                <strong>Auto-sync</strong>
                <p className="muted">On reconnect, on pending changes, and every 2 minutes</p>
              </div>
            </div>
            <div className="conn-row">
              <button className={`toggle ${status?.pull_all ? "on" : ""}`} onClick={togglePullAll}>
                <span className="knob" />
              </button>
              <div>
                <strong>{status?.pull_all ? "Mirror everything" : "Tiered pull"}</strong>
                <p className="muted">
                  {status?.pull_all
                    ? "Every shared memory from every device is copied here"
                    : "Topics, summaries, documents and events come down; other devices' notes stay in the cloud"}
                </p>
              </div>
            </div>
            <label>
              Qdrant Server URL
              <div className="url-row">
                <input
                  value={urlDraft}
                  onChange={(e) => setUrlDraft(e.target.value)}
                  placeholder="http://localhost:6333"
                />
                <button className="ghost" onClick={saveUrl}>
                  Save
                </button>
              </div>
            </label>
            <label>
              API key (Qdrant Cloud)
              <div className="url-row">
                <input
                  type="password"
                  value={keyDraft}
                  onChange={(e) => setKeyDraft(e.target.value)}
                  placeholder={status?.has_api_key ? "A key is saved" : "Not needed for a local server"}
                  autoComplete="off"
                />
                <button className="ghost" onClick={() => saveKey(false)} disabled={!keyDraft.trim()}>
                  Save
                </button>
                {status?.has_api_key && (
                  <button className="ghost" onClick={() => saveKey(true)}>
                    Clear
                  </button>
                )}
              </div>
            </label>
            {status?.device_id && <p className="muted device-id">This device: {status.device_id}</p>}
          </div>

          <div className="sync-card">
            <h3>Memory state</h3>
            <div className="state-grid">
              <div className="state-cell">
                <span className="sc-num pending">{c?.pending ?? "—"}</span>
                <span className="sc-label">pending (outbox)</span>
              </div>
              <div className="state-cell">
                <span className="sc-num synced">{c?.synced ?? "—"}</span>
                <span className="sc-label">synced</span>
              </div>
              <div className="state-cell">
                <span className="sc-num conflict">{c?.conflict ?? "—"}</span>
                <span className="sc-label">conflict</span>
              </div>
              <div className="state-cell">
                <span className="sc-num local">{c?.local_only ?? "—"}</span>
                <span className="sc-label">on device only</span>
              </div>
            </div>
          </div>
        </div>

        <div className="sync-card policy-card">
          <h3>
            Sync policy
            {policy && (
              <span className="muted policy-totals">
                {policy.shared} may sync · {policy.local} stay on this device
              </span>
            )}
          </h3>
          {cats.length === 0 ? (
            <p className="muted">Nothing sensitive detected. Everything may sync.</p>
          ) : (
            <div className="policy-cats">
              {cats.map(([cat, n]) => (
                <span key={cat} className={`policy-chip ${cat}`}>
                  {CATEGORY_LABEL[cat] ?? cat} <b>{n}</b>
                </span>
              ))}
            </div>
          )}
          {policy && policy.recent_local.length > 0 && (
            <ul className="policy-list">
              {policy.recent_local.map((it) => (
                <li key={it.id}>
                  <div>
                    <strong>{it.title}</strong>
                    <span className="muted">{it.reason}</span>
                  </div>
                  {it.category !== "derived" && (
                    <button className="ghost" onClick={() => allowSync(it.id)}>
                      Allow sync
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="sync-actions">
          <button className="primary" onClick={runSync} disabled={busy || !online}>
            {busy ? "Syncing…" : "Sync now"}
          </button>
          <span className="muted">Last sync: {fmtTime(status?.last_sync ?? null)}</span>
        </div>

        {error && <div className="error-bar">{error}</div>}
        {report && <div className="sync-report">{reportLine(report)}</div>}

        <div className="sync-card activity-card">
          <h3>Activity</h3>
          {!status || status.history.length === 0 ? (
            <p className="muted">No syncs yet.</p>
          ) : (
            <ul className="activity-list">
              {status.history.map((h, i) => (
                <li key={i} className={h.ok ? "" : "failed"}>
                  <span className="act-time">{fmtTime(h.at)}</span>
                  <span className="act-trigger">{h.trigger}</span>
                  <span className="act-detail">
                    {h.ok && h.report ? reportLine(h.report) : h.error}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <details className="sync-help">
          <summary>Run a local Qdrant Server</summary>
          <div className="how-body">
            <p>Start Docker Desktop, then run:</p>
            <pre className="how-diagram">docker run -p 6333:6333 qdrant/qdrant</pre>
            <p>
              Or point the URL above at a Qdrant Cloud instance. Toggle airplane mode to
              simulate losing connectivity: capture memories offline (they queue as pending),
              then go online; auto-sync drains the outbox as soon as the server is reachable.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
}
