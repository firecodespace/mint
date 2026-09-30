// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
import { useEffect, useRef, useState } from "react";
import { deleteDocument, ingestDocument, listDocuments } from "../api";
import type { Memory } from "../types";

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export function Vault({
  memories,
  onChange,
}: {
  memories: Memory[];
  onChange: () => void;
}) {
  const [docs, setDocs] = useState<Memory[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function loadDocs() {
    try {
      setDocs(await listDocuments());
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadDocs();
  }, [memories.length]);

  async function ingestFiles(files: File[]) {
    setError(null);
    for (const file of files) {
      setBusy(file.name);
      try {
        const b64 = await fileToBase64(file);
        await ingestDocument(file.name, b64);
      } catch (e) {
        setError(`${file.name}: ${String(e)}`);
      }
    }
    setBusy(null);
    await loadDocs();
    onChange();
  }

  function chunkCount(docId: string) {
    return memories.filter((m) => m.parent_id === docId).length;
  }

  return (
    <div className="vault">
      <div
        className={`dropzone ${dragOver ? "over" : ""} ${busy ? "busy" : ""}`}
        onClick={() => !busy && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!busy) ingestFiles(Array.from(e.dataTransfer.files));
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) ingestFiles(Array.from(e.target.files));
            e.target.value = "";
          }}
        />
        {busy ? (
          <>
            <div className="spinner small" />
            <p>Parsing, chunking and embedding “{busy}”…</p>
          </>
        ) : (
          <>
            <p className="dz-title">Drop documents here, or click to browse</p>
            <p className="muted">
              Text, Markdown, code, and PDF. Parsed, chunked, embedded, and linked into
              memory — all on-device.
            </p>
          </>
        )}
      </div>

      {error && <div className="error-bar">{error}</div>}

      <div className="doc-list">
        <h3>{docs.length} documents</h3>
        {docs.length === 0 && <p className="muted">No documents ingested yet.</p>}
        {docs.map((d) => (
          <div key={d.id} className="doc">
            <div className="doc-main">
              <span className="pill document">document</span>
              <strong>{d.title}</strong>
              <span className="doc-meta">{chunkCount(d.id)} chunks</span>
            </div>
            <div className="doc-row2">
              <span className="doc-when">{new Date(d.created_at).toLocaleString()}</span>
              <button
                className="ghost danger"
                onClick={async () => {
                  await deleteDocument(d.id);
                  await loadDocs();
                  onChange();
                }}
              >
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
