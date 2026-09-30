// Shared TS mirror of the Rust `Memory` record and command payloads.
// Keep in sync with src-tauri/src/memory/record.rs

export type MemoryKind =
  | "observation"
  | "note"
  | "doc_chunk"
  | "measurement"
  | "event"
  | "document"
  | "entity"
  | "summary"
  | "topic";

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
  parent_id: string | null;
  topic_id: string | null;
  archived: boolean;
  due_at: string | null;
  done: boolean;
  /** Why this memory may (or may not) leave the device. */
  sync_reason: string;
  /** Version chain: the older memory this one replaces / the newer one that replaced it. */
  supersedes: string | null;
  superseded_by: string | null;
  /** Device that created this memory. */
  origin: string;
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

// ---- chat ----------------------------------------------------------------

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatStatus {
  ollama_up: boolean;
  models: string[];
  chat_model: string;
  fast_model: string;
}

export interface RetrievedItem {
  id: string;
  title: string;
  kind: string;
  score: number;
  /** "device" (on-device memory) or "cloud" (another device, via cloud search). */
  source: "device" | "cloud";
}

export interface CapturedItem {
  id: string;
  title: string;
  kind: string;
}

export interface ChatTurnResult {
  answer: string;
  retrieved: RetrievedItem[];
  captured: CapturedItem[];
}

// Tauri event payloads emitted during a chat turn.
export interface StageEvent {
  stage: string; // retrieving | retrieved | thinking | answering | extracting | done
  detail: string;
}

export interface TokenEvent {
  channel: "thinking" | "answer";
  text: string;
}

// ---- conversations -------------------------------------------------------

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  ts: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: StoredMessage[];
  memory_ids: string[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
  memory_count: number;
}

// ---- graph & vault -------------------------------------------------------

export interface GraphNode {
  id: string;
  label: string;
  kind: string;
  parent_id: string | null;
  topic_id: string | null;
  salience: number;
  /** Replaced by a newer version. */
  superseded: boolean;
  /** Kept on this device by the sync policy. */
  local_only: boolean;
}

/** An organizing subject (schema node) with its rolling summary. */
export interface TopicInfo {
  id: string;
  name: string;
  summary: string;
  members: number;
  user_named: boolean;
  updated_at: string;
}

export interface OrganizeReport {
  routed: number;
  topics: number;
  summarized: number;
}

export interface MaintenanceReport {
  summaries: number;
  archived: number;
  active: number;
  topics_summarized: number;
  topics_rehomed: number;
}

export interface GraphEdge {
  from: string;
  to: string;
  relation: string; // "related" | "part_of" | "mentions" | "in_topic"
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Memories decayed into the archive (forgotten), excluded from the graph. */
  archived: number;
}

export interface DocIngestResult {
  id: string;
  title: string;
  chunks: number;
}

// ---- sync ----------------------------------------------------------------

export interface SyncCounts {
  pending: number;
  synced: number;
  conflict: number;
  local_only: number;
}

export interface SyncReport {
  pushed: number;
  pulled: number;
  /** Edits on BOTH sides since the last sync; the losing edit is kept as an older version. */
  conflicts: number;
  /** Kept on this device by the sync policy. */
  withheld: number;
  withheld_by: Record<string, number>;
  /** Cloud copies removed because the memory became private. */
  retracted: number;
  /** Other devices' raw memories left in the cloud (tiered pull). */
  cloud_only: number;
}

export interface SyncEvent {
  at: string;
  trigger: string;
  ok: boolean;
  report: SyncReport | null;
  error: string | null;
}

export interface SyncDecision {
  share: boolean;
  category: string;
  reason: string;
}

export interface PolicyItem {
  id: string;
  title: string;
  category: string;
  reason: string;
}

export interface PolicySummary {
  shared: number;
  local: number;
  local_by_category: Record<string, number>;
  recent_local: PolicyItem[];
}

export interface SyncStatus {
  online: boolean;
  reachable: boolean;
  server_url: string;
  counts: SyncCounts;
  last_sync: string | null;
  last_report: SyncReport | null;
  auto: boolean;
  pull_all: boolean;
  device_id: string;
  history: SyncEvent[];
  /** An API key is configured (the key itself is never sent to the UI). */
  has_api_key: boolean;
}
