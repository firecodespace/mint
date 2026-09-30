// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
import { deleteMemory } from "../api";
import type { Memory } from "../types";

export function Browser({
  memories,
  onDeleted,
  onError,
}: {
  memories: Memory[];
  onDeleted: (id: string) => void;
  onError: (msg: string) => void;
}) {
  async function remove(id: string) {
    try {
      await deleteMemory(id);
      onDeleted(id);
    } catch (e) {
      onError(String(e));
    }
  }

  return (
    <div className="browser">
      <h2>Device memory</h2>
      <p className="hint">{memories.length} memories stored on this device.</p>

      {memories.length === 0 && (
        <p className="muted">Nothing captured yet. Add an observation from the Capture tab.</p>
      )}

      <ul className="mem-list">
        {memories.map((m) => (
          <li key={m.id} className="mem">
            <div className="mem-top">
              <span className={`pill ${m.kind}`}>{m.kind}</span>
              <strong>{m.title || "(untitled)"}</strong>
              <span className={`sync ${m.sync_state}`}>{m.sync_state}</span>
              <button className="ghost danger" onClick={() => remove(m.id)} title="Delete">
                Remove
              </button>
            </div>
            <p className="mem-text">{m.text}</p>
            <div className="mem-meta">
              {m.site_id && <span>site: {m.site_id}</span>}
              {m.asset_id && <span>asset: {m.asset_id}</span>}
              {m.sensitivity === "local_only" && <span className="lock">local-only</span>}
              {m.tags.map((t) => (
                <span key={t} className="tag">
                  #{t}
                </span>
              ))}
              <span className="when">{new Date(m.captured_at).toLocaleString()}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
