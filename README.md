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

## Development notes

- **Always build from `src-tauri/`** (`cargo` runs there, and `npm run tauri dev`
  invokes cargo there). `mint-core` compiles into `src-tauri/target` as a path
  dependency. Do **not** run `cargo` from `crates/mint-core` or add a workspace root:
  a new target directory produces fresh build-script binaries that Windows Smart App
  Control blocks (`os error 4551`). The `src-tauri/target` location is trusted.
- **`MINT_DATA_DIR`** overrides where memory is stored. In dev we set it to
  `D:\mint\.mint-data` (project-local, easy to inspect). Unset -> per-user app data dir.
- **`MINT_CHAT_MODEL` / `MINT_FAST_MODEL`** override the Ollama models (defaults:
  `qwen3:8b` for chat with visible thinking, `llama3.2:latest` for extraction).
- Requires a local **Ollama** server on `localhost:11434` with a model pulled.

## Status
Phase 2 (cognitive core: `mint-core` + Ollama chat + memory capture) — in progress.
See the roadmap in `SPEC.md`.

_Xarch Labs_
