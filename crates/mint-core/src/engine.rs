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
use super::meta::MetaStore;
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

#[derive(Debug, Clone, Serialize, Default)]
pub struct MaintenanceReport {
    pub summaries: usize,
    pub archived: usize,
    pub active: usize,
}

/// Decay thresholds.
const DECAY_SALIENCE: f32 = 0.30;
const DECAY_MIN_AGE_DAYS: f64 = 14.0;
/// Consolidation: minimum memories mentioning an entity to summarize it, and cap.
const CONSOLIDATE_MIN: usize = 3;
const CONSOLIDATE_MAX_ENTITIES: usize = 12;

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
    meta: Mutex<MetaStore>,
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
        let meta = MetaStore::open(data_dir)?;

        let engine = Self {
            shard,
            embedders,
            graph: Mutex::new(graph),
            meta: Mutex::new(meta),
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
            archived: false,
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

    // ---- sync (edge <-> cloud) ------------------------------------------

    pub fn sync_counts(&self) -> Result<SyncCounts> {
        let mut c = SyncCounts::default();
        for m in self.list()? {
            match m.sync_state {
                SyncState::Pending => c.pending += 1,
                SyncState::Synced => c.synced += 1,
                SyncState::Conflict => c.conflict += 1,
                SyncState::LocalOnly => c.local_only += 1,
            }
        }
        Ok(c)
    }

    fn to_server_point(&self, m: &Memory) -> Result<ServerPoint> {
        let (dense, sparse) = self.embedders.embed_document(&m.text)?;
        Ok(ServerPoint {
            id: uuid_string(&m.id)?,
            dense,
            sparse_indices: sparse.indices,
            sparse_values: sparse.values,
            payload: serde_json::to_value(m)?,
        })
    }

    /// Two-way sync with a Qdrant Server. Local-only and entity nodes never
    /// leave the device. Divergences resolve last-write-wins by `updated_at`.
    pub fn sync(&self, client: &SyncClient) -> Result<SyncReport> {
        client.ensure_collection(DENSE_DIM)?;

        // Remote state, keyed by memory id (from payloads).
        let mut remote: HashMap<String, Memory> = HashMap::new();
        for p in client.scroll_all()? {
            if let Ok(m) = serde_json::from_value::<Memory>(p) {
                remote.insert(m.id.clone(), m);
            }
        }

        // Local syncable memories (shareable, non-entity).
        let local: HashMap<String, Memory> = self
            .list()?
            .into_iter()
            .filter(|m| m.sensitivity == Sensitivity::Shareable && m.kind != MemoryKind::Entity)
            .map(|m| (m.id.clone(), m))
            .collect();

        let mut report = SyncReport::default();
        let mut to_push: Vec<Memory> = Vec::new();
        let mut to_pull: Vec<Memory> = Vec::new();

        let mut ids: std::collections::HashSet<&String> = local.keys().collect();
        ids.extend(remote.keys());

        for id in ids {
            match (local.get(id), remote.get(id)) {
                (Some(l), None) => to_push.push(l.clone()),
                (None, Some(r)) => to_pull.push(r.clone()),
                (Some(l), Some(r)) => {
                    if l.updated_at == r.updated_at {
                        // Already in agreement; mark synced if still pending.
                        if l.sync_state == SyncState::Pending {
                            let mut m = l.clone();
                            m.sync_state = SyncState::Synced;
                            let _ = self.upsert(&m);
                        }
                    } else {
                        // Divergence: last write wins, but flag it as a conflict
                        // that was auto-resolved.
                        report.conflicts += 1;
                        if l.updated_at > r.updated_at {
                            to_push.push(l.clone());
                        } else {
                            to_pull.push(r.clone());
                        }
                    }
                }
                (None, None) => {}
            }
        }

        // Push: mark synced first so both the server payload and the local copy
        // agree on sync_state, then upsert to server and to the local shard.
        if !to_push.is_empty() {
            let synced: Vec<Memory> = to_push
                .iter()
                .map(|m| {
                    let mut mm = m.clone();
                    mm.sync_state = SyncState::Synced;
                    mm
                })
                .collect();
            let points: Vec<ServerPoint> = synced
                .iter()
                .filter_map(|m| self.to_server_point(m).ok())
                .collect();
            client.upsert(&points)?;
            for mm in &synced {
                if self.upsert(mm).is_ok() {
                    report.pushed += 1;
                }
            }
        }

        // Pull (store remote memory locally, marked synced).
        for r in &to_pull {
            let mut mm = r.clone();
            mm.sync_state = SyncState::Synced;
            if self.upsert(&mm).is_ok() {
                report.pulled += 1;
            }
        }

        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(report)
    }

    pub fn graph_data(&self) -> Result<GraphData> {
        let memories = self.list()?;
        let active: std::collections::HashSet<String> = memories
            .iter()
            .filter(|m| !m.archived)
            .map(|m| m.id.clone())
            .collect();
        let nodes: Vec<GraphNode> = memories
            .iter()
            .filter(|m| !m.archived)
            .map(|m| GraphNode {
                id: m.id.clone(),
                label: if m.title.trim().is_empty() {
                    m.text.chars().take(28).collect()
                } else {
                    m.title.clone()
                },
                kind: kind_str(&m.kind),
                parent_id: m.parent_id.clone(),
                salience: m.salience,
            })
            .collect();
        let edges = self
            .graph
            .lock()
            .map(|g| g.all().to_vec())
            .unwrap_or_default()
            .into_iter()
            .filter(|e| active.contains(&e.from) && active.contains(&e.to))
            .collect();
        Ok(GraphData { nodes, edges })
    }

    // ---- consolidation, decay, maintenance ------------------------------

    /// Distill what is known about well-connected entities into summary nodes.
    pub fn consolidate(&self, model: &str) -> Result<usize> {
        let edges = self
            .graph
            .lock()
            .map(|g| g.all().to_vec())
            .unwrap_or_default();

        // entity id -> memories that mention it.
        let mut ent_mems: HashMap<String, Vec<String>> = HashMap::new();
        for e in &edges {
            if e.relation == "mentions" {
                ent_mems.entry(e.to.clone()).or_default().push(e.from.clone());
            }
        }

        let all = self.list()?;
        let by_id: HashMap<String, Memory> =
            all.iter().map(|m| (m.id.clone(), m.clone())).collect();

        let mut ents: Vec<(String, Vec<String>)> = ent_mems
            .into_iter()
            .filter(|(_, v)| v.len() >= CONSOLIDATE_MIN)
            .collect();
        ents.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        ents.truncate(CONSOLIDATE_MAX_ENTITIES);

        let mut count = 0;
        for (ent_id, mem_ids) in ents {
            let ent = match by_id.get(&ent_id) {
                Some(e) => e,
                None => continue,
            };
            let mut notes = String::new();
            for mid in &mem_ids {
                if let Some(m) = by_id.get(mid) {
                    if m.kind == MemoryKind::Summary {
                        continue;
                    }
                    notes.push_str("- ");
                    notes.push_str(&truncate(&m.text, 300));
                    notes.push('\n');
                }
            }
            if notes.trim().is_empty() {
                continue;
            }
            let prompt = format!(
                "Entity: {}\nNotes mentioning it:\n{}\nWrite a concise factual summary \
(3-5 sentences) of what is known about \"{}\" from these notes only.",
                ent.title, notes, ent.title
            );
            let summary = match self.ollama.generate(
                model,
                Some("You write concise factual entity summaries for a knowledge base."),
                &prompt,
                false,
            ) {
                Ok(s) if !s.trim().is_empty() => s,
                _ => continue,
            };

            let existing = self.meta.lock().ok().and_then(|m| m.summary_for(&ent_id));
            let title = format!("{} — summary", ent.title);
            let summary_id = if let Some(sid) = existing.filter(|sid| by_id.contains_key(sid)) {
                // Update the existing summary node in place.
                let mut updated = by_id.get(&sid).unwrap().clone();
                updated.text = summary;
                updated.title = title;
                updated.updated_at = chrono::Utc::now().to_rfc3339();
                let _ = self.upsert(&updated);
                sid
            } else {
                let s = self.add(NewMemory {
                    kind: MemoryKind::Summary,
                    title,
                    text: summary,
                    site_id: String::new(),
                    asset_id: String::new(),
                    geo: None,
                    tags: vec![ent.title.clone(), "summary".to_string()],
                    source: MemorySource::Manual,
                    sensitivity: Sensitivity::Shareable,
                    parent_id: None,
                })?;
                let _ = self
                    .meta
                    .lock()
                    .map(|mut m| m.set_summary(&ent_id, &s.id));
                s.id
            };
            // Attach the summary to its entity hub.
            if let Ok(mut g) = self.graph.lock() {
                let _ = g.set_mentions(&summary_id, &[ent_id.clone()]);
            }
            count += 1;
        }
        Ok(count)
    }

    /// Archive old, low-salience document chunks (kept, excluded from retrieval).
    pub fn decay(&self) -> Result<usize> {
        let all = self.list()?;
        let mut newly = 0;
        for m in &all {
            if m.archived || m.kind != MemoryKind::DocChunk {
                continue;
            }
            if m.salience < DECAY_SALIENCE && age_days(&m.updated_at) > DECAY_MIN_AGE_DAYS {
                if let Ok(mut meta) = self.meta.lock() {
                    if meta.archive(&m.id).is_ok() {
                        newly += 1;
                    }
                }
            }
        }
        Ok(newly)
    }

    /// Full maintenance pass: consolidate + decay + prune stale archive entries.
    pub fn run_maintenance(&self, model: &str) -> Result<MaintenanceReport> {
        let summaries = self.consolidate(model)?;
        let _ = self.decay()?;
        let existing: std::collections::HashSet<String> =
            self.list()?.iter().map(|m| m.id.clone()).collect();
        if let Ok(mut meta) = self.meta.lock() {
            let _ = meta.retain_existing(&existing);
        }
        let archived = self
            .meta
            .lock()
            .map(|m| m.archived_ids().len())
            .unwrap_or(0);
        let active = self.list_active()?.len();
        Ok(MaintenanceReport {
            summaries,
            archived,
            active,
        })
    }

    pub fn clear_archive(&self) -> Result<usize> {
        self.meta
            .lock()
            .map_err(|_| anyhow!("meta poisoned"))?
            .clear_archive()
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

        let archived = self
            .meta
            .lock()
            .map(|m| m.archived_ids().clone())
            .unwrap_or_default();
        let results: Vec<SearchResult> = scored
            .into_iter()
            .filter_map(|sp| {
                let memory = memory_from_payload(sp.payload.as_ref())?;
                if archived.contains(&memory.id) {
                    return None;
                }
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

        // Overlay engine-computed state (archived flag + salience) that lives
        // outside the vector payload so it never triggers re-embedding.
        let archived = self
            .meta
            .lock()
            .map(|m| m.archived_ids().clone())
            .unwrap_or_default();
        let deg = self.degrees();
        for m in &mut memories {
            m.archived = archived.contains(&m.id);
            m.salience = salience_of(m, deg.get(&m.id).copied().unwrap_or(0));
        }

        // Newest first by ULID (lexicographically sortable, time-ordered).
        memories.sort_by(|a, b| b.id.cmp(&a.id));
        Ok(memories)
    }

    /// All non-archived memories.
    pub fn list_active(&self) -> Result<Vec<Memory>> {
        Ok(self.list()?.into_iter().filter(|m| !m.archived).collect())
    }

    /// Connectivity degree per node id (edges touching it).
    fn degrees(&self) -> HashMap<String, usize> {
        let mut d = HashMap::new();
        if let Ok(g) = self.graph.lock() {
            for e in g.all() {
                *d.entry(e.from.clone()).or_insert(0) += 1;
                *d.entry(e.to.clone()).or_insert(0) += 1;
            }
        }
        d
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

/// Truncate to at most `max` characters (char-safe).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect::<String>() + "..."
    }
}

/// Age of an RFC3339 timestamp in days.
fn age_days(ts: &str) -> f64 {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|t| {
            (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_seconds() as f64 / 86400.0
        })
        .unwrap_or(0.0)
}

/// Salience in [0,1]: recency + graph connectivity + kind weight.
fn salience_of(m: &Memory, degree: usize) -> f32 {
    let recency = (1.0 / (1.0 + age_days(&m.updated_at) / 30.0)) as f32;
    let deg = (degree as f32 / 4.0).min(1.0);
    let kindw = match m.kind {
        MemoryKind::Note
        | MemoryKind::Observation
        | MemoryKind::Event
        | MemoryKind::Measurement
        | MemoryKind::Summary => 1.0,
        MemoryKind::Entity => 0.9,
        MemoryKind::Document => 0.85,
        MemoryKind::DocChunk => 0.5,
    };
    0.5 * recency + 0.3 * deg + 0.2 * kindw
}

/// Deterministic Qdrant point id from a ULID string (UUID over the 16 ULID bytes).
fn point_id(ulid_str: &str) -> Result<PointId> {
    let ulid = ulid::Ulid::from_string(ulid_str).context("invalid ULID")?;
    let uuid = qdrant_edge::external::uuid::Uuid::from_bytes(ulid.to_bytes());
    Ok(PointId::Uuid(uuid))
}

/// Same derivation as `point_id`, as a hyphenated UUID string for the server.
fn uuid_string(ulid_str: &str) -> Result<String> {
    let ulid = ulid::Ulid::from_string(ulid_str).context("invalid ULID")?;
    let uuid = qdrant_edge::external::uuid::Uuid::from_bytes(ulid.to_bytes());
    Ok(uuid.to_string())
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
