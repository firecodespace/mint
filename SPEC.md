# Mint — AI-Powered Edge Memory & Intelligence Platform

> A local-first, offline-capable AI platform that remembers. Everything you tell it,
> write, commit, or drop in becomes searchable semantic memory on-device; a local LLM
> (Ollama) reasons over that memory, keeps it organized, and syncs intelligently to a
> cloud brain when connected.
>
> Powered by **Qdrant Edge** (embedded vector search) + **fastembed** (local embeddings)
> + **Ollama** (local reasoning), inside a **Tauri** desktop app, with a **`mint-core`**
> engine exposed over a **localhost API** so other surfaces (CLI, git hooks, IDE plugin)
> are all clients.

Edge-native successor to **HCMA** — the cognitive-memory idea, actually built.

---

## 0. Product decisions (locked 2026-09-29)

- **Qdrant Edge challenge stays in scope.** Edge<->cloud sync + conflict resolution +
  a sync-status UI remain required deliverables (Phase 5). The cloud is reframed as
  backup / cross-device sharing of *consolidated* memory, which also serves the product.
- **`mint-core` extraction happens now** (start of Phase 2): the engine becomes a
  library hosted by a local service that exposes a **localhost-only API**. The Tauri app
  is a client; the CLI, git hooks, and VS Code plugin are clients too. Everything stays
  offline (localhost).
- **Phase 2 leads with the general chat-memory assistant** — a local chatbot that
  remembers everything you tell it and shows its work. Developer tooling is Phase 4.
- **No emoji / pictographs anywhere in the codebase.** Text labels and inline SVG only.

---

## 1. The unifying idea: one event bus, one cognitive core

Every input — chat turns, dropped docs/images, git commits/PRs, `#mem` code comments,
`mint add` from the CLI, "I have a deadline Friday" — is the same primitive: a
**MemoryEvent** entering a pipeline. Build one cognitive core; everything else is an
**ingestion adapter** or a **surface**.

```
  INGESTION ADAPTERS            COGNITIVE CORE (mint-core, 100% local)          SURFACES
  ─────────────────            ────────────────────────────────────           ────────
  chat turns          ┐        1. normalize   -> MemoryEvent                   Chat console
  dropped docs/images ├──────► 2. extract     (Ollama: entities/tasks/dates)   + Cognition panel
  git hooks (commit/PR)│       3. embed        (fastembed dense + BM25 sparse)  + Plan-flow timeline
  #mem comments / .mint│       4. route/salience (store? merge? discard? secret)Dev dashboard
  CLI `mint add`      ┘        5. store/update (Qdrant Edge, layered)           VS Code plugin
                              6. consolidate   (Ollama: summarize/dedup/decay)  (all CLIENTS of
                              7. propose action (task/date -> confirm)           the localhost API)
                                 ▲
                              retrieval (hybrid + temporal/graph) feeds chat + dashboards
```

Ollama is the reasoning organ used at steps 2, 6, 7, and generation — that is what makes
this *cognitive*, not a vector DB with a chat skin.

---

## 2. Architecture (target)

```
mint/
├── crates/
│   ├── mint-core/     Rust lib: memory, embed, ollama client, pipeline, retrieval, sync
│   └── mint-daemon/   (optional later) headless host of mint-core + localhost API
├── src-tauri/         Tauri app: hosts mint-core, serves the UI, exposes localhost API
├── src/               React UI: chat + cognition panel + plan-flow + dev dashboard
├── cli/               `mint` CLI (Phase 4): git hooks, `mint add`, capture
└── vscode/            VS Code extension (Phase 4): client of the localhost API
```

- **Localhost API**: an `axum` server inside the app (or daemon) bound to `127.0.0.1`,
  offline only. Endpoints for capture, search, chat (SSE stream), tasks, sync status.
- **Same core, many clients**: UI, CLI, git hooks, IDE plugin all speak to the API.

---

## 3. Cognitive memory model

Reviving HCMA's layered memory — built for real this time:

- **Episodic** — raw events (a chat turn, a commit, a captured note). High volume.
- **Semantic** — consolidated facts / understanding, distilled from episodics by Ollama.
- **Procedural** — how-tos / recurring patterns (later).

A **consolidation pass** (Ollama, on-demand/background) merges episodic -> semantic,
updates existing memories (the HCMA "understanding score" idea), dedups, and decays stale
low-salience memories. This is the "auto-update / better organization" goal.

Each memory keeps the Phase-1 fields plus: `layer`, `salience`, `sensitivity`,
`sync_state`, `version`, and links (source event, related memories, backlinks to file:line
or a chat turn).

### Secrets are special (security)
Never embed secret values. Store secret **metadata** semantically (name, service,
purpose, where-used, rotation date) for retrieval; keep the **value** in the OS credential
vault (Windows DPAPI / Credential Manager) and store only a reference. Search finds the
metadata; the value never lands in the vector store or an embedding.

---

## 4. Surfaces & UI (no emoji)

Three-zone cognition layout:
- **Center** — chat.
- **Right rail "Cognition"** — streams Ollama's live reasoning for the current turn.
- **Bottom "Plan flow"** — real-time timeline of pipeline stages firing
  (`extracting -> embedding -> routing -> stored #id`, `proposed task`,
  `consolidated 3 memories`) so the user watches it think and save, locally.
- **Separate route "Developer dashboard"** — git worklog, recent captures, secrets
  metadata, open tasks. Purpose-built, uncluttered. (Phase 4.)

---

## 5. Developer capture strategy (Phase 4)

Explicit + git, never a file-save firehose (noise wrecks retrieval):
- **git hooks** (`post-commit`, `pre-push`, PR via `gh`) -> auto worklog memory
  (what changed, why, files/branch, Ollama summary).
- **`#mem` / `#todo` inline comment tags** -> ingested with `file:line` backlink.
- **`.mint` scratch file** watched for new lines.
- **CLI `mint add "..."`** and IDE "capture selection".
Salience routing keeps the store signal-rich.

---

## 6. Stack

- Tauri 2 (Rust core + React/Vite/TS UI).
- Qdrant Edge 0.8 (embedded, dense + sparse hybrid, on-disk).
- fastembed 7 (all-MiniLM-L6-v2, 384d; offline after first run).
- Ollama (local LLM: chat, extraction, consolidation, proposals; multimodal for images).
- axum for the localhost API.
- Cloud sync target (Phase 5): Qdrant Server (Docker / Qdrant Cloud).

---

## 7. Roadmap

- **Phase 1 — Retrieval engine.** DONE. Offline hybrid search (dense + sparse + RRF),
  payload filters, disk persistence, capture/search/browse UI, activity log.
- **Phase 2 — Cognitive core + Chat + Ollama.** Extract `mint-core` + localhost API;
  Ollama streaming chat with RAG over memory; auto-capture memories from conversation
  (extract -> embed -> route -> store); new 3-zone UI; purge all emoji.
- **Phase 3 — Intelligence.** Layered memory (episodic/semantic/procedural),
  consolidation / auto-update / dedup / decay, salience routing, multimodal ingest
  (docs + images via local captioning), proactive task/date extraction -> local
  tasks/calendar with confirm.
- **Phase 4 — Developer platform.** `mint` CLI + git hooks, `#mem` / `.mint` capture,
  VS Code extension, developer dashboard.
- **Phase 5 — Edge<->cloud sync brain.** Qdrant Server sync + conflict resolution +
  sync-status UI (challenge deliverable; also cross-device memory). Connectivity
  detector + airplane-mode toggle + outbox + sensitivity gate.

---

## 8. Status log
- 2026-09-28: Toolchain + Tauri 2 scaffold. Phase 1 retrieval engine built and working
  (Qdrant Edge + fastembed hybrid search, on-disk persistence). Data dir relocated to
  project-local `D:\mint\.mint-data` via `MINT_DATA_DIR`.
- 2026-09-29: Platform vision locked (see section 0). Next: Phase 2 — start with the
  `mint-core` refactor + localhost API, then Ollama chat with live memory capture.
- 2026-09-30: Topic (schema) layer: kNN + entity-aware routing, Profile topic,
  consolidation (re-homing), rolling LLM summaries, topic-aware retrieval with
  weighted RRF, rename/merge/move/organize from the graph. Labeled benchmark harness
  and integration tests added; chat retrieval MRR 0.660 -> 1.000, topic F1 0.733 ->
  0.944. Batched ingestion (3x per-chunk), indexed lookups, optimized dev deps.
  Still open: localhost API, CLI/git hooks, secrets vault, procedural memory, images.
