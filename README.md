# Mint — Edge Memory for Field Agents

An **offline-first** desktop app powered by **Qdrant Edge**. A disconnected field agent
captures observations, searches them instantly with **zero network** via an embedded
in-process vector engine, and (Phase 2) intelligently syncs curated knowledge to a
shared cloud brain when connectivity returns.

Edge-native successor to **HCMA**. See [`SPEC.md`](./SPEC.md) for the full design.

## Stack
- **Tauri 2** (Rust core + React/Vite/TS UI) — one binary, in-process vector engine.
- **Qdrant Edge 0.8** — embedded vector search (dense + sparse hybrid), on-disk.
- **fastembed 7** — local ONNX embeddings (`bge-small-en-v1.5`), offline after first run.
- **Ollama** — local LLM reasoning (Phase 3).

## Prerequisites
- Node 18+, Rust (stable-msvc), MSVC C++ build tools, WebView2 (Windows).

## Run (dev)
```bash
npm install
npm run tauri dev
```
> First build compiles Qdrant Edge + fastembed and downloads the embedding model once
> (needs network that one time). Offline forever after.

## Layout
```
src/          React frontend (memory browser, search playground, activity log)
src-tauri/    Rust core
  src/memory/ record.rs · embed.rs (fastembed) · engine.rs (Qdrant Edge)
  src/commands.rs   Tauri commands bridging UI <-> core
```

## Status
Phase 1 (retrieval engine) — in progress. See the roadmap in `SPEC.md`.

_Xarch Labs_
