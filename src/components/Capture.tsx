import { useState } from "react";
import { addMemory } from "../api";
import type { Memory, MemoryKind, NewMemory, Sensitivity } from "../types";

const KINDS: MemoryKind[] = ["observation", "note", "measurement", "event", "doc_chunk"];

export function Capture({
  onAdded,
  onError,
}: {
  onAdded: (m: Memory) => void;
  onError: (msg: string) => void;
}) {
  const [kind, setKind] = useState<MemoryKind>("observation");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [siteId, setSiteId] = useState("");
  const [assetId, setAssetId] = useState("");
  const [tags, setTags] = useState("");
  const [sensitivity, setSensitivity] = useState<Sensitivity>("shareable");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    const input: NewMemory = {
      kind,
      title: title.trim(),
      text: text.trim(),
      site_id: siteId.trim(),
      asset_id: assetId.trim(),
      tags: tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
      source: "manual",
      sensitivity,
    };
    try {
      const m = await addMemory(input);
      onAdded(m);
      setTitle("");
      setText("");
      setTags("");
    } catch (err) {
      onError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="capture" onSubmit={submit}>
      <h2>Capture observation</h2>
      <p className="hint">
        Stored on-device and embedded locally. No network required.
      </p>

      <div className="grid2">
        <label>
          Kind
          <select value={kind} onChange={(e) => setKind(e.target.value as MemoryKind)}>
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label>
          Sensitivity
          <select
            value={sensitivity}
            onChange={(e) => setSensitivity(e.target.value as Sensitivity)}
          >
            <option value="shareable">shareable</option>
            <option value="local_only">local_only (never syncs)</option>
          </select>
        </label>
      </div>

      <label>
        Title
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Short label" />
      </label>

      <label>
        Observation
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="What did you observe? e.g. Coolant leak at pump P-12, pressure dropping…"
          rows={5}
          required
        />
      </label>

      <div className="grid3">
        <label>
          Site
          <input value={siteId} onChange={(e) => setSiteId(e.target.value)} placeholder="site-a" />
        </label>
        <label>
          Asset
          <input value={assetId} onChange={(e) => setAssetId(e.target.value)} placeholder="P-12" />
        </label>
        <label>
          Tags
          <input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="leak, urgent" />
        </label>
      </div>

      <button className="primary" type="submit" disabled={busy}>
        {busy ? "Embedding…" : "Capture to edge memory"}
      </button>
    </form>
  );
}
