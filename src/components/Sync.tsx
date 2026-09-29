import { useCallback, useEffect, useState } from "react";
import { setOnline, setServerUrl, syncNow, syncStatus } from "../api";
import type { SyncReport, SyncStatus } from "../types";

function fmtTime(epochSecs: string | null): string {
  if (!epochSecs) return "never";
  const n = Number(epochSecs);
  if (!Number.isFinite(n)) return epochSecs;
  return new Date(n * 1000).toLocaleTimeString();
}

export function Sync({ onChange }: { onChange: () => void }) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [urlDraft, setUrlDraft] = useState("");
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

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  async function toggleOnline() {
    if (!status) return;
    await setOnline(!status.online);
    await refresh();
  }

  async function saveUrl() {
    await setServerUrl(urlDraft.trim());
    await refresh();
  }

  async function runSync() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const r = await syncNow();
      setReport(r);
      await refresh();
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

  return (
    <div className="sync-tab">
      <div className="sync-inner">
        <h1>Edge &lt;-&gt; Cloud sync</h1>
        <p className="how-lead">
          Everything runs offline on this device. When you are online and a Qdrant Server
          is reachable, shareable memories sync to the cloud and updates come back down.
          Local-only memories never leave the device.
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
                <span className="sc-label">local-only</span>
              </div>
            </div>
          </div>
        </div>

        <div className="sync-actions">
          <button className="primary" onClick={runSync} disabled={busy || !online}>
            {busy ? "Syncing…" : "Sync now"}
          </button>
          <span className="muted">Last sync: {fmtTime(status?.last_sync ?? null)}</span>
        </div>

        {error && <div className="error-bar">{error}</div>}
        {report && (
          <div className="sync-report">
            Pushed <strong>{report.pushed}</strong> · Pulled <strong>{report.pulled}</strong> ·
            Conflicts resolved <strong>{report.conflicts}</strong>
          </div>
        )}

        <details className="sync-help">
          <summary>Run a local Qdrant Server</summary>
          <div className="how-body">
            <p>Start Docker Desktop, then run:</p>
            <pre className="how-diagram">docker run -p 6333:6333 qdrant/qdrant</pre>
            <p>
              Or point the URL above at a Qdrant Cloud instance. Toggle airplane mode to
              simulate losing connectivity: capture memories offline (they queue as
              pending), then go online and Sync now to drain the outbox.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
}
