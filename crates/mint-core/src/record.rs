//! The on-device unit of memory. Serializes 1:1 into the Qdrant Edge point
//! payload, so a round-trip (upsert -> query/scroll -> deserialize) reconstructs
//! the full record. Mirrored in the frontend at src/types.ts.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Observation,
    Note,
    DocChunk,
    Measurement,
    Event,
    /// Root node of an ingested document (its chunks link to it via parent_id).
    Document,
    /// A named entity (person/org/place/concept) that connects memories.
    Entity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    Manual,
    File,
    Sensor,
    /// Auto-captured from a chat conversation.
    Chat,
}

impl MemoryKind {
    /// Parse a free-form kind string (from an LLM), defaulting to `Note`.
    pub fn parse_lenient(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "observation" => MemoryKind::Observation,
            "measurement" => MemoryKind::Measurement,
            "event" => MemoryKind::Event,
            "doc_chunk" | "docchunk" | "document" => MemoryKind::DocChunk,
            _ => MemoryKind::Note,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    Shareable,
    LocalOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SyncState {
    LocalOnly,
    Pending,
    Synced,
    Conflict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Geo {
    pub lat: f64,
    pub lng: f64,
}

/// A single memory. `id` is a ULID string; the Qdrant point id is a UUID
/// derived deterministically from the ULID's 16 bytes (see engine.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub kind: MemoryKind,
    pub title: String,
    pub text: String,

    // Field-agent structured fields (payload-indexed for fast filtering).
    pub site_id: String,
    pub asset_id: String,
    pub geo: Option<Geo>,
    pub tags: Vec<String>,
    pub source: MemorySource,

    pub captured_at: String, // RFC3339
    pub created_at: String,
    pub updated_at: String,

    // Engine-managed.
    pub salience: f32,
    pub sensitivity: Sensitivity,
    pub sync_state: SyncState,
    pub version: u64,

    /// For document chunks: the id of their parent Document node.
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Input for creating a memory. The engine fills id / timestamps / the
/// engine-managed fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    pub kind: MemoryKind,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub site_id: String,
    #[serde(default)]
    pub asset_id: String,
    #[serde(default)]
    pub geo: Option<Geo>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub source: MemorySource,
    pub sensitivity: Sensitivity,
    #[serde(default)]
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Dense,
    Sparse,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub limit: usize,
    #[serde(default)]
    pub site_id: Option<String>,
    #[serde(default)]
    pub kind: Option<MemoryKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memory: Memory,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub latency_ms: f64,
    pub mode: SearchMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub total: usize,
    pub by_kind: std::collections::HashMap<String, usize>,
}
