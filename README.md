# Mint — AI-Powered Edge Memory & Intelligence Platform

Mint is an **offline-first** AI platform that remembers. You chat with a local language
model; everything you tell it is answered on-device and distilled into searchable
semantic memory. Embeddings, vector search, and reasoning all run locally — nothing
leaves the machine, and it keeps working with no network.

Powered by **Qdrant Edge** (embedded vector search) + **fastembed** (local embeddings) +
**Ollama** (local reasoning), inside a **Tauri** desktop app. Edge-native successor to
**HCMA** — the cognitive-memory idea, actually built. See [`SPEC.md`](./SPEC.md) for the
full design and roadmap.

---

## The problem

AI at the edge must search and reason over locally generated information without
depending on the cloud: latency matters, connectivity is intermittent, and sensitive
data cannot always leave the device. The goal is an offline-first app that maintains
local semantic memory, retrieves it with low latency, keeps working offline, and
synchronizes intelligently when a connection returns.

## The approach: one event bus, one cognitive core

Every input — a chat turn, a dropped document, a git commit, a captured note — is the
same primitive: a **memory event** flowing through a local pipeline. One cognitive core
processes them all; everything else is an ingestion adapter or a display surface.

```
chat / docs / git / notes
      |
      v
  normalize  ->  extract (LLM)  ->  embed (dense + sparse)
      ->  route / dedup  ->  store & update (Qdrant Edge)
      ->  consolidate (LLM)  ->  retrieve (hybrid)  ->  answer
```

The language model is not a chatbot bolted on — it is the reasoning organ used to
extract memories, consolidate them, and generate answers. That is what makes Mint
*cognitive* rather than a vector database with a chat skin.

## The stack (all local)

| Layer | Technology |
|---|---|
| Vector store | **Qdrant Edge 0.8** — embedded, in-process, on-disk. Dense + sparse hybrid search. No server. |
| Embeddings | **fastembed 7** — ONNX all-MiniLM-L6-v2 (384d). Downloaded once, then offline. |
| Reasoning | **Ollama** — local LLM (default `qwen3:8b` for chat with visible thinking, `llama3.2` for extraction). |
| App | **Tauri 2** (Rust core) + **React** UI. |
| Engine | **`mint-core`** — a reusable Rust library (memory, embeddings, Ollama, chat, conversations). |

## How retrieval works

A query is embedded two ways: a **dense** vector (semantic meaning) and a **sparse**
vector (BM25 keywords). Qdrant Edge runs both searches and fuses them with Reciprocal
Rank Fusion, so meaning-based and exact-word matches come back together — typically in a
few milliseconds, with no network.

## What becomes a memory (and what does not)

After each turn a fast model extracts only **new, durable facts you stated** — your name,
decisions, preferences, plans, deadlines. Questions, greetings, and restatements are
ignored. Before storing, a similarity check skips near-duplicates, so asking about
something you already told Mint does not create a second memory.

## What is delivered

- **Offline retrieval engine** — hybrid (dense + sparse + RRF) search, payload filters,
  on-disk persistence that survives restarts.
- **Local chat** — streamed answer and streamed model *thinking* (Cognition panel), with
  a live pipeline timeline (Plan flow).
- **Automatic memory capture** — durable facts distilled from conversation, deduplicated.
- **Conversations** — persistent chat threads with a sidebar; deleting a chat can
  optionally purge the memories it created.
- **Memory graph** — an Obsidian-style force-directed view where memories branch off
  shared tags and kinds, plus a searchable list view.
- **How it works** — an in-app explanation of the whole system.

## What is next

- **Intelligence** — layered memory (episodic/semantic/procedural), consolidation and
  decay, multimodal ingest (documents and images), proactive tasks and dates.
- **Developer platform** — a localhost API, a `mint` CLI with git hooks, inline `#mem`
  capture, and a VS Code plugin.
- **Sync** — edge-to-cloud synchronization to a Qdrant Server with conflict resolution
  and a sync-status view, for backup and cross-device memory.

---

## Run (dev)

Prerequisites: Node 18+, Rust (stable-msvc), MSVC C++ build tools, WebView2 (Windows),
and a local **Ollama** server on `localhost:11434` with a model pulled
(`ollama pull qwen3:8b`).

```bash
npm install
npm run tauri dev
```

The first launch compiles the engine and downloads the embedding model once (needs
network that one time); after that it runs fully offline.

## How memory is organized (topics)

Every document and note is filed under a **topic** — the schema layer that groups a
research effort, project, or theme. Routing is a k-nearest-neighbour vote: memories
already filed and similar to the new content vote for their topic, along with the
topic's own rolling summary and shared entities (e.g. two notes that both mention
"I-20" and "DSO"). Identity documents (resume / CV) get a dedicated Profile topic.
Maintenance re-homes stragglers once their subject exists, and a local LLM keeps a
rolling summary per topic. You can rename, merge, and move topics from the graph.

Retrieval is topic-aware: hybrid search finds the evidence, the subjects it belongs to
are activated (the subject of the *best* evidence wins), their members are pulled in,
named documents ("my resume", "coresum", "HCMA") are injected, and everything is fused
with weighted reciprocal-rank fusion. Chat receives the subject overview first, then
the individual memories.

## Edge <-> cloud: what stays, what syncs, how conflicts resolve

**Sync policy (decided per memory, with a stated reason).** A deterministic, on-device
scan runs on every write and again at sync time. Secrets (passwords, API keys,
tokens, private keys), financial details (Luhn-checked card numbers, bank/IBAN
numbers), government IDs, personal health facts (first-person only, so a research
paper about cancer still syncs), and contact details stay on the device. Entity hubs
are derived and rebuilt per device. Topic summaries are built only from shareable
members, and a topic whose members are all private stays private, so nothing leaks
through a summary. You can override any decision; making something private retracts
its cloud copy.

**Evolving and conflicting information.** When a new fact revises an older one ("my
exam moved to Monday"), the older memory is marked superseded and chained to the new
one (deterministic rules first, then the local chat model as judge). Retrieval answers
from the current version, the timeline hides outdated deadlines, and the graph shows
the version history. Sync uses a 3-way merge against the last-synced version: if only
one side changed, it wins cleanly; if both changed, the newer edit wins and the other
is preserved as an older version, so no edit is ever lost.

**Tiered edge-to-cloud memory.** Each device pushes what the policy allows. It pulls
consolidated knowledge (topics, summaries, documents, events) and its own memories;
other devices' raw notes stay in the cloud and are reached through cloud search when
online (short timeout, labeled "cloud" in chat). Offline, everything works locally.
Auto-sync runs on reconnect, on pending changes, and every two minutes, with an
activity log in the Sync view.

## Benchmarks and tests

A labeled benchmark lives in `crates/mint-core/examples/bench`: documents and notes
across several subjects, 20 graded queries (vocabulary mismatch, personal questions,
named documents, topic-scoped questions), capture-guard cases, and optional distractor
documents including hard negatives. It reports Recall@6, MRR, nDCG@6, Hit@1,
precision, latency, ingestion throughput, and topic-routing purity (including whether
a distractor topic absorbed a real subject). Results are saved to
`crates/mint-core/bench-results/`.

```bash
cd crates/mint-core
cargo run --release --example bench -- --label mine --compare baseline
cargo run --release --example bench -- --label scale --scale 150
cargo run --release --example bench -- --label llm --llm llama3.2:latest
cargo test --release
```

Current results vs. the original pipeline (chat retrieval, local LLM mode):

| Metric | Before | Now |
|---|---|---|
| MRR | 0.660 | 1.000 |
| Hit@1 | 0.50 | 1.00 |
| nDCG@6 | 0.677 | 0.925 |
| Recall@6 | 0.912 | 0.975 |
| p50 latency | 7.8 ms | 4.7 ms |
| Topic pairwise F1 | 0.733 | 0.944 (0 impure topics with 150 distractors) |
| Capture-guard accuracy | 0.917 | 1.000 |

Edge <-> cloud (added with the sync policy and version chains):

| Check | Result |
|---|---|
| Sync policy, 34 labeled cases | precision 1.000, recall 1.000, category accuracy 1.000 |
| Sync policy false positives on the research corpus | 0 of 26 items |
| Version chains, 14 cases (rules only) | precision 1.000, recall 0.750 |
| Version chains, 14 cases (rules + qwen3:8b judge) | precision 1.000, recall 0.875; current version ranked first 1.000 |
| Two-device sync against a real Qdrant Server | 7 of 7 scenarios pass (privacy, retraction, tiered sharing + cloud search, one-sided edits, concurrent edits kept as versions, deletions, offline) |

The sync tests (`cargo test --release --test sync`) use throwaway collections on the
server at `QDRANT_URL` (default `localhost:6333`) and skip if none is running.

## Development notes

- **Always build from `src-tauri/`** (`npm run tauri dev` does this). `mint-core`
  compiles into `src-tauri/target` as a path dependency. Do **not** run `cargo` from
  `crates/mint-core` or add a Cargo workspace root: a new target directory produces
  fresh build-script binaries that Windows Smart App Control blocks (`os error 4551`).
  The `src-tauri/target` location is trusted.
- **`MINT_DATA_DIR`** overrides where memory is stored (dev: `D:\mint\.mint-data`,
  project-local and easy to inspect). Unset -> per-user app data dir.
- **`MINT_CHAT_MODEL` / `MINT_FAST_MODEL`** override the Ollama models.

## Layout

```
crates/mint-core/   Rust engine: record, embed, engine (Qdrant Edge),
                    ollama, chat, conversations
src-tauri/          Tauri app: commands + state, hosts mint-core
src/                React UI: Chat (+ sidebar, cognition, plan-flow),
                    MemoryGraph, Search/Capture/Browser, HowItWorks
```

_Xarch Labs_
