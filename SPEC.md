# Mint — Edge Memory for Field Agents

> An offline-first desktop app where a disconnected field agent captures observations,
> searches them instantly with **zero network** via embedded **Qdrant Edge**, and
> intelligently syncs curated knowledge to a shared cloud brain (**Qdrant Server**)
> when connectivity returns.

Built as the edge-native successor to **HCMA** (Hierarchical Cognitive Memory Architecture).

---

## 1. Problem & Goal

AI at the edge must search and reason over locally generated information without a
constant cloud dependency: robots, kiosks, vehicles, mobile and field devices operate
where connectivity is intermittent, latency matters, and sensitive data cannot always
leave the device.

**Goal:** an offline-first AI application, powered by Qdrant Edge, that can:

1. Maintain searchable semantic memory directly on the device.
2. Perform low-latency vector + hybrid search with no network access.
3. Operate fully offline and survive intermittent connectivity.
4. Dynamically decide what stays local vs. what is synchronized.
5. Synchronize with Qdrant Server when connectivity returns.
6. Handle evolving memory, updates, and conflicting information.
7. Provide a UI to inspect device memory, search results, sync status, and activity.

**Expected outcome:** a complete edge-native AI product that can *remember, retrieve,
operate offline, and sync intelligently when connected* — not merely a local vector DB.

---

## 2. What we carry over from HCMA (and what we drop)

| HCMA (cloud-heavy) | Mint (edge-native) |
|---|---|
| Qdrant **Server** (`:6333`) as primary store | Qdrant **Edge** — embedded in-process in the Tauri Rust core, offline |
| Python FastAPI service for embeddings | **fastembed** (Rust, ONNX) in-process — dense + sparse, no network |
| Neo4j graph + Redis cache | Dropped. Flat memory records + payload filters (graph edges optional later) |
| Ollama `scoreSummaries` (70% LLM / 20% recency / 10% length) | **Salience/sync brain** reuses the same weighted-scorer shape |
| Stable `ctx_${nodeId}_${userId}` IDs + Cypher `MERGE` | Stable ULIDs + idempotent upsert (survives conflict / replay) |
| Gemini for final answer | Local **Ollama** answer/summary — Phase 3, optional, offline |

The transferable HCMA insight: **the LLM is one signal among cheap heuristics in a
weighted scorer**, not the whole decision. In HCMA that decided which understanding
summary "won". In Mint it decides **how salient a memory is and whether it should be
promoted to the cloud**.

---

## 3. Target platform & stack

- **App shell:** Tauri 2 (Rust core + web UI). One binary, in-process vector engine,
  genuinely offline. Target: Windows desktop (dev machine), portable to other OSes.
- **Vector engine:** `qdrant-edge` 0.8, embedded in the Rust core.
- **Embeddings:** `fastembed` 7.x (Rust) — dense (`bge-small-en-v1.5`, 384d) + sparse
  (SPLADE / BM25) for true hybrid search. Model cached on disk after first download →
  offline thereafter.
- **Frontend:** React 19 + Vite + TypeScript, talking to the Rust core via Tauri commands.
- **Cloud sync target (Phase 2):** Qdrant Server (Docker locally, or Qdrant Cloud).
- **Local LLM (Phase 3):** Ollama (already installed: v0.15.5).

### Why Tauri over Electron / Python
Qdrant Edge is a Rust crate that embeds **in-process**. Tauri's Rust core hosts it
directly — no sidecar server, no separate process — which is the honest "edge" story.

---

## 4. Data model — the Memory record

The on-device unit of memory.

```
Memory {
  id: ULID                       # stable, idempotent upsert key
  kind: observation | note | doc_chunk | measurement | event
  title: string
  text: string                   # the content that gets embedded

  # field-agent structured fields (payload-indexed for fast filtering)
  site_id: string
  asset_id: string
  geo: { lat: f64, lng: f64 } | null
  tags: string[]
  source: manual | file | sensor

  captured_at: timestamp         # when the observation happened
  created_at: timestamp
  updated_at: timestamp

  # engine-managed
  salience: f32                  # Phase 2 (heuristic) -> Phase 3 (Ollama-blended)
  sensitivity: local_only | shareable
  sync_state: local_only | pending | synced | conflict
  version: u64                   # for conflict resolution
}
```

Stable ULID + idempotent upsert = the HCMA "stable ID / MERGE" property: re-ingesting
or replaying a sync never duplicates a memory.

---

## 5. Qdrant Edge collection design

- **Named vectors:** `dense` (cosine) + `sparse` → hybrid retrieval.
- **Payload indexes:** `kind`, `tags`, `site_id`, `asset_id`, `captured_at`, `sync_state`
  → fast filtered search ("leaks at Site A last week").
- **Persistence:** shard persisted to disk (app data dir) so memory survives restart.

---

## 6. Retrieval flow (Phase 1 core)

```
query text
  -> fastembed: dense embedding + sparse embedding   (in-process, no network)
  -> Qdrant Edge query:
       prefetch dense (topK) + prefetch sparse (topK)
       fuse with RRF (Reciprocal Rank Fusion)
       apply payload filters (site/date/kind/tags)
  -> results + payload + score
  -> UI shows results AND measured latency (proves low-latency, no network)
```

### Ingestion
- **Note / observation:** embed text → upsert directly.
- **Dropped document (PDF/txt):** recursive text-splitting in Rust (no LLM yet) →
  embed each chunk → upsert as `doc_chunk`. Semantic chunking via Ollama is Phase 3.

---

## 7. The sync brain (Phase 2 — the differentiator)

- **Connectivity detector** + in-app **"airplane mode" toggle** for live demos.
- **Outbox / op-log:** every write made offline is journaled locally.
- **Salience scoring (heuristics first):** access frequency, freshness, confidence,
  kind. `sensitivity = local_only` gate → private/device-specific memories never sync.
- **Sync policy:** on reconnect → drain outbox up to Qdrant Server, pull cloud updates
  down, keyed by stable ULID.
- **Conflict resolution:** same ULID edited on device + cloud → compare `version` →
  Last-Write-Wins or merge → surfaced in the UI.

---

## 8. Local reasoning (Phase 3 — last)

- **Ollama as the salience judge** — the HCMA `scoreSummaries` pattern.
- **Offline Q&A** over retrieved memories (RAG, fully local).
- **Semantic chunking** upgrade for document ingestion.

---

## 9. UI surfaces

1. **Memory browser** — list/inspect on-device memories, filter by site/kind/tags.
2. **Search playground** — query box, dense/sparse/hybrid toggle, filters, results with
   **visible latency**.
3. **Sync dashboard** — online/offline toggle, outbox depth, last sync, conflicts. (P2)
4. **Activity log** — a stream of system events (ingest, search, sync, conflict).

---

## 10. Phased roadmap

### Phase 1 — Retrieval engine (PRIORITY)
- [x] Tauri + React/Vite scaffold.
- [ ] `Memory` record type (Rust + shared TS types).
- [ ] Qdrant Edge shard: named dense+sparse vectors, payload indexes, disk persist.
- [ ] fastembed dense + sparse embedding module.
- [ ] Ingest: add note; drop a document -> chunk -> embed -> upsert.
- [ ] Hybrid search (RRF) + payload filters, Tauri command.
- [ ] UI: memory browser + search playground with latency readout.
- [ ] Durability: restart app, memory persists.

### Phase 2 — Sync brain
### Phase 3 — Local Ollama reasoning

---

## 11. Status log
- 2026-09-28: Toolchain ready (Node 22, Rust 1.98.1 msvc, MSVC, WebView2, Ollama 0.15.5).
  Tauri 2 + React-TS scaffolded. Deps added: qdrant-edge 0.8, fastembed 7.1, ulid,
  chrono, anyhow. Building Phase 1 retrieval core.
