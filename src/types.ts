// Shared TS mirror of the Rust `Memory` record and command payloads.
// Keep in sync with src-tauri/src/memory/record.rs

export type MemoryKind =
  | "observation"
  | "note"
  | "doc_chunk"
  | "measurement"
  | "event";

export type MemorySource = "manual" | "file" | "sensor";

export type Sensitivity = "shareable" | "local_only";

export type SyncState = "local_only" | "pending" | "synced" | "conflict";

export interface Geo {
  lat: number;
  lng: number;
}

export interface Memory {
  id: string;
  kind: MemoryKind;
  title: string;
  text: string;
  site_id: string;
  asset_id: string;
  geo: Geo | null;
  tags: string[];
  source: MemorySource;
  captured_at: string; // RFC3339
  created_at: string;
  updated_at: string;
  salience: number;
  sensitivity: Sensitivity;
  sync_state: SyncState;
  version: number;
}

// Input for creating a memory (engine fills id/timestamps/engine-managed fields).
export interface NewMemory {
  kind: MemoryKind;
  title: string;
  text: string;
  site_id: string;
  asset_id: string;
  tags: string[];
  source: MemorySource;
  sensitivity: Sensitivity;
}

export type SearchMode = "dense" | "sparse" | "hybrid";

export interface SearchRequest {
  query: string;
  mode: SearchMode;
  limit: number;
  site_id?: string | null;
  kind?: MemoryKind | null;
}

export interface SearchResult {
  memory: Memory;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
  latency_ms: number;
  mode: SearchMode;
}

export interface Stats {
  total: number;
  by_kind: Record<string, number>;
}
