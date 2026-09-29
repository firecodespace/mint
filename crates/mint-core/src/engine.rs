//! The on-device memory engine: an embedded Qdrant Edge shard plus local
//! embedders. All operations are synchronous, in-process, and offline.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use qdrant_edge::{
    Condition, CreateIndex, Distance, EdgeConfigBuilder, EdgeShard, EdgeSparseVectorParamsBuilder,
    EdgeVectorParamsBuilder, FieldCondition, FieldIndexOperations, Filter, Fusion, JsonPath, Match,
    Modifier, NamedQuery, Payload, PayloadFieldSchema, PayloadSchemaType, PointId, PointOperations,
    PointInsertOperations, PointStruct, PointStructPersisted, PrefetchBuilder, QueryEnum,
    QueryRequestBuilder, ScoringQuery, ScrollRequestBuilder, UpdateOperation, Vector, VectorInternal,
    Vectors, WithPayloadInterface,
};

use super::embed::{Embedders, DENSE_DIM};
use super::graph::{GraphData, GraphNode, GraphStore};
use super::ollama::Ollama;
use super::record::{
    Memory, MemoryKind, MemorySource, NewMemory, SearchMode, SearchRequest, SearchResponse,
    SearchResult, Sensitivity, Stats, SyncState,
};
use super::sync::{ServerPoint, SyncClient, SyncReport};
use super::{documents, entities};

use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct SyncCounts {
    pub pending: usize,
    pub synced: usize,
    pub conflict: usize,
    pub local_only: usize,
}

/// Similarity above which two memories get a "related" edge in the graph.
const RELATED_THRESHOLD: f32 = 0.55;
/// Max related edges recorded per memory.
const RELATED_MAX: usize = 4;

const DENSE_NAME: &str = "dense";
const SPARSE_NAME: &str = "sparse";
const EDGE_CONFIG_FILE: &str = "edge_config.json";
/// How many candidates each arm of a hybrid search pulls before fusion.
const PREFETCH_LIMIT: usize = 50;
/// Upper bound when scrolling all points (Phase 1: local, modest data sizes).
const SCROLL_ALL_LIMIT: usize = 10_000;

pub struct MemoryEngine {
    shard: EdgeShard,
    embedders: Embedders,
    graph: Mutex<GraphStore>,
    ollama: Ollama,
    /// normalized entity name -> entity memory id (dedup).
    entity_index: Mutex<HashMap<String, String>>,
}

impl MemoryEngine {
    /// Open (or create) the engine under `data_dir`. Layout:
    ///   <data_dir>/shard   -> Qdrant Edge shard (persisted memory)
    ///   <data_dir>/models  -> fastembed ONNX cache (downloaded once)
    pub fn open(data_dir: &Path) -> Result<Self> {
        let shard_dir = data_dir.join("shard");
        let models_dir = data_dir.join("models");
        std::fs::create_dir_all(&shard_dir).context("failed to create shard dir")?;

        let embedders = Embedders::new(&models_dir)?;
        let shard = Self::open_shard(&shard_dir)?;
        let graph = GraphStore::open(data_dir)?;

        let engine = Self {
            shard,
            embedders,
            graph: Mutex::new(graph),
            ollama: Ollama::new(),
            entity_index: Mutex::new(HashMap::new()),
        };
        engine.ensure_indexes();
        engine.rebuild_entity_index();
        Ok(engine)
    }

    /// Rebuild the entity name -> id map from existing Entity nodes.
    fn rebuild_entity_index(&self) {
        if let Ok(all) = self.list() {
            if let Ok(mut idx) = self.entity_index.lock() {
                idx.clear();
                for m in all.iter().filter(|m| m.kind == MemoryKind::Entity) {
                    idx.insert(m.title.trim().to_lowercase(), m.id.clone());
                }
            }
        }
    }

    fn open_shard(shard_dir: &Path) -> Result<EdgeShard> {
        let dense = EdgeVectorParamsBuilder::new(DENSE_DIM, Distance::Cosine).build();
        // BM25 sparse with IDF applied at query time (Qdrant's recommended setup).
        let sparse = EdgeSparseVectorParamsBuilder::new()
            .modifier(Modifier::Idf)
            .build();
        let config = EdgeConfigBuilder::new()
            .on_disk_payload(true)
            .vector(DENSE_NAME, dense)
            .sparse_vector(SPARSE_NAME, sparse)
            .build();

        let exists = shard_dir.join(EDGE_CONFIG_FILE).exists();
        if exists {
            EdgeShard::load(shard_dir, Some(config))
                .map_err(|e| anyhow!("failed to load edge shard: {e}"))
        } else {
            EdgeShard::new(shard_dir, config)
                .map_err(|e| anyhow!("failed to create edge shard: {e}"))
        }
    }

    /// Create keyword/datetime payload indexes for the fields we filter on.
    /// Idempotent across runs; re-creating an existing index is ignored.
    fn ensure_indexes(&self) {
        let keyword_fields = ["kind", "site_id", "asset_id", "tags", "sync_state", "source"];
        for field in keyword_fields {
            self.create_index(field, PayloadSchemaType::Keyword);
        }
        self.create_index("captured_at", PayloadSchemaType::Datetime);
    }

    fn create_index(&self, field: &str, schema: PayloadSchemaType) {
        let op = UpdateOperation::FieldIndexOperation(FieldIndexOperations::CreateIndex(
            CreateIndex {
                field_name: jpath(field),
                field_schema: Some(PayloadFieldSchema::FieldType(schema)),
            },
        ));
        if let Err(e) = self.shard.update(op) {
            log::debug!("index create for '{field}' skipped: {e}");
        }
    }

    // ---- ingestion -------------------------------------------------------

    /// Create a new memory from user/agent input, embed and store it.
    pub fn add(&self, input: NewMemory) -> Result<Memory> {
        let now = chrono::Utc::now().to_rfc3339();
        let id = ulid::Ulid::generate().to_string();
        // Local-only memories and entity hubs never sync; everything else is
        // queued as pending for the next sync.
        let sync_state = if input.sensitivity == Sensitivity::LocalOnly
            || input.kind == MemoryKind::Entity
        {
            SyncState::LocalOnly
        } else {
            SyncState::Pending
        };
        let memory = Memory {
            id,
            kind: input.kind,
            title: input.title,
            text: input.text,
            site_id: input.site_id,
            asset_id: input.asset_id,
            geo: input.geo,
            tags: input.tags,
            source: input.source,
            captured_at: now.clone(),
            created_at: now.clone(),
            updated_at: now,
            salience: 0.0,
            sensitivity: input.sensitivity,
            sync_state,
            version: 1,
            parent_id: input.parent_id,
        };
        self.upsert(&memory)?;
        self.shard
            .flush()
            .map_err(|e| anyhow!("flush failed: {e}"))?;
        // Establish semantic relationships (skip chunks: they connect via
        // part_of; skip entities: they are hubs connected via mentions).
        if memory.kind != MemoryKind::DocChunk && memory.kind != MemoryKind::Entity {
            self.link_related(&memory);
        }
        Ok(memory)
    }

    // ---- entities --------------------------------------------------------

    /// Find an existing entity node by name, or create one. Deduped by name.
    fn find_or_create_entity(&self, name: &str) -> Result<String> {
        let key = name.trim().to_lowercase();
        if key.is_empty() {
            return Err(anyhow!("empty entity name"));
        }
        if let Ok(idx) = self.entity_index.lock() {
            if let Some(id) = idx.get(&key) {
                return Ok(id.clone());
            }
        }
        let entity = self.add(NewMemory {
            kind: MemoryKind::Entity,
            title: name.trim().to_string(),
            text: name.trim().to_string(),
            site_id: String::new(),
            asset_id: String::new(),
            geo: None,
            tags: vec!["entity".to_string()],
            source: MemorySource::Manual,
            sensitivity: Sensitivity::Shareable,
            parent_id: None,
        })?;
        if let Ok(mut idx) = self.entity_index.lock() {
            idx.insert(key, entity.id.clone());
        }
        Ok(entity.id)
    }

    /// Extract entities from `text` and link `from_id` to them via "mentions".
    /// Returns the entity ids linked.
    pub fn attach_entities(&self, from_id: &str, text: &str, model: &str) -> Result<Vec<String>> {
        let names = entities::extract(&self.ollama, model, text)?;
        let mut ids = Vec::new();
        for name in names {
            if let Ok(id) = self.find_or_create_entity(&name) {
                if id != from_id {
                    ids.push(id);
                }
            }
        }
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.set_mentions(from_id, &ids);
        }
        Ok(ids)
    }

    /// Find this memory's nearest neighbours and record "related" edges.
    fn link_related(&self, memory: &Memory) {
        let hits = match self.search(SearchRequest {
            query: memory.text.clone(),
            mode: SearchMode::Dense,
            limit: RELATED_MAX + 3,
            site_id: None,
            kind: None,
        }) {
            Ok(r) => r.results,
            Err(_) => return,
        };
        let tos: Vec<String> = hits
            .into_iter()
            .filter(|r| r.memory.id != memory.id && r.score >= RELATED_THRESHOLD)
            .take(RELATED_MAX)
            .map(|r| r.memory.id)
            .collect();
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.set_related(&memory.id, &tos);
        }
    }

    /// Add a memory only if no near-duplicate already exists (dense cosine).
    /// Returns `None` when a memory with similarity >= `threshold` is present.
    /// Used for auto-capture so restated facts do not pile up.
    pub fn add_if_novel(&self, input: NewMemory, threshold: f32) -> Result<Option<Memory>> {
        let hits = self.search(SearchRequest {
            query: input.text.clone(),
            mode: SearchMode::Dense,
            limit: 1,
            site_id: None,
            kind: None,
        })?;
        if let Some(top) = hits.results.first() {
            if top.score >= threshold {
                return Ok(None);
            }
        }
        Ok(Some(self.add(input)?))
    }

    /// Insert or update a memory by its stable id (idempotent upsert).
    pub fn upsert(&self, mem: &Memory) -> Result<()> {
        let (dense, sparse) = self.embedders.embed_document(&mem.text)?;
        let pid = point_id(&mem.id)?;
        let vectors = Vectors::new_named(vec![
            (DENSE_NAME, Vector::new_dense(dense)),
            (SPARSE_NAME, Vector::from(sparse)),
        ]);
        let payload = serde_json::to_value(mem).context("failed to serialize memory payload")?;
        let point = PointStruct::new(pid, vectors, payload);
        let points: Vec<PointStructPersisted> = vec![point.into()];

        self.shard
            .update(UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::PointsList(points)),
            ))
            .map_err(|e| anyhow!("upsert failed: {e}"))?;
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let pid = point_id(id)?;
        self.shard
            .update(UpdateOperation::PointOperation(
                PointOperations::DeletePoints { ids: vec![pid] },
            ))
            .map_err(|e| anyhow!("delete failed: {e}"))?;
        self.shard
            .flush()
            .map_err(|e| anyhow!("flush failed: {e}"))?;
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.remove_node(id);
        }
        Ok(())
    }

    // ---- documents (Vault) ----------------------------------------------

    /// Parse + chunk + store a document as a Document node with chunk children.
    /// `entity_model` (when non-empty) is used to link the document to entities.
    /// Returns the document node and the number of chunks stored.
    pub fn ingest_document(
        &self,
        filename: &str,
        bytes: &[u8],
        entity_model: &str,
    ) -> Result<(Memory, usize)> {
        let text = documents::parse(filename, bytes)?;
        if text.trim().is_empty() {
            return Err(anyhow!("no extractable text in '{filename}'"));
        }
        let preview: String = text.chars().take(400).collect();
        let doc = self.add(NewMemory {
            kind: MemoryKind::Document,
            title: filename.to_string(),
            text: preview,
            site_id: String::new(),
            asset_id: String::new(),
            geo: None,
            tags: Vec::new(),
            source: MemorySource::File,
            sensitivity: Sensitivity::Shareable,
            parent_id: None,
        })?;

        let chunks = documents::chunk_text(&text);
        let n = chunks.len();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let stored = self.add(NewMemory {
                kind: MemoryKind::DocChunk,
                title: format!("{filename} [{}]", i + 1),
                text: chunk,
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags: Vec::new(),
                source: MemorySource::File,
                sensitivity: Sensitivity::Shareable,
                parent_id: Some(doc.id.clone()),
            })?;
            if let Ok(mut g) = self.graph.lock() {
                let _ = g.add_part_of(&stored.id, &doc.id);
            }
        }
        // Link the document to the entities it mentions (best-effort).
        if !entity_model.is_empty() {
            let _ = self.attach_entities(&doc.id, &text, entity_model);
        }
        Ok((doc, n))
    }

    pub fn list_documents(&self) -> Result<Vec<Memory>> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|m| m.kind == MemoryKind::Document)
            .collect())
    }

    /// Delete a document node and all of its chunks.
    pub fn delete_document(&self, doc_id: &str) -> Result<()> {
        let children: Vec<String> = self
            .list()?
            .into_iter()
            .filter(|m| m.parent_id.as_deref() == Some(doc_id))
            .map(|m| m.id)
            .collect();
        for c in children {
            let _ = self.delete(&c);
        }
        self.delete(doc_id)
    }

    // ---- graph -----------------------------------------------------------

    pub fn graph_data(&self) -> Result<GraphData> {
        let memories = self.list()?;
        let nodes: Vec<GraphNode> = memories
            .iter()
            .map(|m| GraphNode {
                id: m.id.clone(),
                label: if m.title.trim().is_empty() {
                    m.text.chars().take(28).collect()
                } else {
                    m.title.clone()
                },
                kind: kind_str(&m.kind),
                parent_id: m.parent_id.clone(),
            })
            .collect();
        let edges = self
            .graph
            .lock()
            .map(|g| g.all().to_vec())
            .unwrap_or_default();
        Ok(GraphData { nodes, edges })
    }

    // ---- retrieval -------------------------------------------------------

    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        let started = Instant::now();
        let (dense, sparse) = self.embedders.embed_query(&req.query)?;
        let filter = build_filter(req.site_id.as_deref(), req.kind.as_ref());

        let mut builder = QueryRequestBuilder::new(req.limit)
            .with_payload(WithPayloadInterface::Bool(true));
        if let Some(f) = filter {
            builder = builder.filter(f);
        }

        builder = match req.mode {
            SearchMode::Dense => builder.query(nearest(DENSE_NAME, VectorInternal::Dense(dense))),
            SearchMode::Sparse => {
                builder.query(nearest(SPARSE_NAME, VectorInternal::Sparse(sparse)))
            }
            SearchMode::Hybrid => builder
                .add_prefetch(
                    PrefetchBuilder::new(PREFETCH_LIMIT)
                        .query(nearest(DENSE_NAME, VectorInternal::Dense(dense)))
                        .build(),
                )
                .add_prefetch(
                    PrefetchBuilder::new(PREFETCH_LIMIT)
                        .query(nearest(SPARSE_NAME, VectorInternal::Sparse(sparse)))
                        .build(),
                )
                .query(ScoringQuery::Fusion(Fusion::Rrf {
                    k: 60,
                    weights: None,
                })),
        };

        let scored = self
            .shard
            .query(builder.build())
            .map_err(|e| anyhow!("query failed: {e}"))?;

        let results: Vec<SearchResult> = scored
            .into_iter()
            .filter_map(|sp| {
                let memory = memory_from_payload(sp.payload.as_ref())?;
                Some(SearchResult {
                    memory,
                    score: sp.score,
                })
            })
            .collect();

        Ok(SearchResponse {
            results,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            mode: req.mode,
        })
    }

    /// All memories, newest first.
    pub fn list(&self) -> Result<Vec<Memory>> {
        let req = ScrollRequestBuilder::new()
            .limit(SCROLL_ALL_LIMIT)
            .with_payload(WithPayloadInterface::Bool(true))
            .build();
        let (records, _next) = self
            .shard
            .scroll(req)
            .map_err(|e| anyhow!("scroll failed: {e}"))?;

        let mut memories: Vec<Memory> = records
            .into_iter()
            .filter_map(|r| memory_from_payload(r.payload.as_ref()))
            .collect();
        // Newest first by ULID (lexicographically sortable, time-ordered).
        memories.sort_by(|a, b| b.id.cmp(&a.id));
        Ok(memories)
    }

    pub fn stats(&self) -> Result<Stats> {
        let memories = self.list()?;
        let mut by_kind = std::collections::HashMap::new();
        for m in &memories {
            let key = kind_str(&m.kind);
            *by_kind.entry(key).or_insert(0) += 1;
        }
        Ok(Stats {
            total: memories.len(),
            by_kind,
        })
    }
}

// ---- helpers -------------------------------------------------------------

/// Build a JsonPath for a payload field (our field names are simple identifiers).
fn jpath(field: &str) -> JsonPath {
    field.parse().expect("valid payload field path")
}

/// Deterministic Qdrant point id from a ULID string (UUID over the 16 ULID bytes).
fn point_id(ulid_str: &str) -> Result<PointId> {
    let ulid = ulid::Ulid::from_string(ulid_str).context("invalid ULID")?;
    let uuid = qdrant_edge::external::uuid::Uuid::from_bytes(ulid.to_bytes());
    Ok(PointId::Uuid(uuid))
}

fn nearest(vector_name: &str, vector: VectorInternal) -> ScoringQuery {
    ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
        query: vector,
        using: Some(vector_name.to_string()),
    }))
}

fn build_filter(site_id: Option<&str>, kind: Option<&MemoryKind>) -> Option<Filter> {
    let mut must: Vec<Condition> = Vec::new();
    if let Some(s) = site_id {
        if !s.is_empty() {
            must.push(Condition::Field(FieldCondition::new_match(
                jpath("site_id"),
                Match::from(s.to_string()),
            )));
        }
    }
    if let Some(k) = kind {
        must.push(Condition::Field(FieldCondition::new_match(
            jpath("kind"),
            Match::from(kind_str(k)),
        )));
    }
    if must.is_empty() {
        None
    } else {
        Some(Filter {
            must: Some(must),
            ..Default::default()
        })
    }
}

fn memory_from_payload(payload: Option<&Payload>) -> Option<Memory> {
    let payload = payload?;
    let value = serde_json::to_value(payload).ok()?;
    serde_json::from_value::<Memory>(value).ok()
}

/// snake_case string for a MemoryKind (via its serde representation).
fn kind_str(kind: &MemoryKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}
