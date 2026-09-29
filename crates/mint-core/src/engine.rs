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
    Modifier, NamedQuery, Payload, PayloadFieldSchema, PayloadOps, PayloadSchemaType, PointId,
    PointInsertOperations, PointOperations, PointStruct, PointStructPersisted, PrefetchBuilder,
    QueryEnum, QueryRequestBuilder, RetrieveRequestBuilder, ScoringQuery, ScrollRequestBuilder,
    SetPayloadOp, CountRequestBuilder,
    UpdateOperation, Vector, VectorInternal, Vectors, WithPayloadInterface,
};

use super::embed::{Embedders, DENSE_DIM};
use super::graph::{GraphData, GraphNode, GraphStore};
use super::meta::MetaStore;
use super::ollama::Ollama;
use super::record::{
    Memory, MemoryKind, MemorySource, NewMemory, SearchMode, SearchRequest, SearchResponse,
    SearchResult, Sensitivity, Stats, SyncState,
};
use super::sync::{ServerPoint, SyncClient, SyncReport, TombstonePoint};
use super::{documents, entities, policy};

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
    /// Topics whose rolling summary was refreshed this pass.
    pub topics_summarized: usize,
    /// Memories re-homed from small topics into established ones.
    pub topics_rehomed: usize,
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
/// Retrieval + organization parameters. Defaults are chosen by the benchmark
/// (`cargo run --release --example bench`); override with `set_tuning`.
#[derive(Debug, Clone, Copy)]
pub struct Tuning {
    /// Min dense similarity of an already-filed neighbour for its topic to get
    /// a routing vote (kNN routing).
    pub topic_join: f32,
    /// Min similarity to a topic node itself (name + rolling summary).
    pub topic_join_direct: f32,
    /// Min share of the topic vote for a topic to activate at query time.
    pub activation_min: f32,
    /// Top-topic share above which off-topic results are demoted.
    pub focus_conf: f32,
    /// Score multiplier for results outside the active topics when focused.
    pub offtopic_penalty: f32,
    /// RRF weight of an activated topic's member list (times its share).
    pub topic_weight: f32,
    /// RRF weight of a document the query names explicitly.
    pub named_weight: f32,
    /// RRF weight of profile documents for questions about the user.
    pub profile_weight: f32,
    /// RRF rank constant (smaller = sharper preference for top ranks).
    pub rrf_k: f32,
    /// Max results from any one document in the final context.
    pub per_doc_cap: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            // Benchmarked at scale (150 distractor docs incl. hard negatives):
            // 0.40 lets a distractor topic absorb a real paper; 0.45 keeps every
            // topic pure with the best grouping; 0.50+ over-splits.
            topic_join: 0.45,
            topic_join_direct: 0.55,
            activation_min: 0.30,
            focus_conf: 0.55,
            offtopic_penalty: 0.6,
            topic_weight: 1.0,
            named_weight: 1.5,
            profile_weight: 1.0,
            rrf_k: 30.0,
            per_doc_cap: 3,
        }
    }
}

/// Whether a memory may leave the device, and why.
#[derive(Debug, Clone, Serialize)]
pub struct SyncDecision {
    pub share: bool,
    /// secret | financial | government_id | health | contact | user | derived
    /// (local) or knowledge | document | memory | user (shared).
    pub category: String,
    pub reason: String,
}

impl SyncDecision {
    fn share(category: &str, reason: &str) -> Self {
        Self { share: true, category: category.into(), reason: reason.into() }
    }
    fn local(category: &str, reason: &str) -> Self {
        Self { share: false, category: category.into(), reason: reason.into() }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyItem {
    pub id: String,
    pub title: String,
    pub category: String,
    pub reason: String,
}

/// Device-level view of the sync policy.
#[derive(Debug, Clone, Serialize, Default)]
pub struct PolicySummary {
    pub shared: usize,
    pub local: usize,
    pub local_by_category: HashMap<String, usize>,
    pub recent_local: Vec<PolicyItem>,
}

/// A topic as shown to the user (legend / management UI).
#[derive(Debug, Clone, Serialize)]
pub struct TopicInfo {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub members: usize,
    pub user_named: bool,
    pub updated_at: String,
}

/// Result of organizing existing (unfiled) memories into topics.
#[derive(Debug, Clone, Serialize, Default)]
pub struct OrganizeReport {
    pub routed: usize,
    pub topics: usize,
    pub summarized: usize,
}

/// Topic activated for a query, with its rolling summary (if any).
#[derive(Debug, Clone, Serialize)]
pub struct TopicHit {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub confidence: f32,
}

/// Everything chat grounding needs: ranked evidence + the subjects it belongs to.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalContext {
    pub results: Vec<SearchResult>,
    pub topics: Vec<TopicHit>,
}

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
    tuning: Tuning,
}

impl MemoryEngine {
    /// Open (or create) the engine under `data_dir`. Layout:
    ///   <data_dir>/shard   -> Qdrant Edge shard (persisted memory)
    ///   <data_dir>/models  -> fastembed ONNX cache (downloaded once)
    pub fn open(data_dir: &Path) -> Result<Self> {
        Self::open_with_models(data_dir, &data_dir.join("models"))
    }

    /// Open with an explicit embedding-model cache dir (lets tests/benchmarks use
    /// a throwaway data dir without re-downloading the model).
    pub fn open_with_models(data_dir: &Path, models_dir: &Path) -> Result<Self> {
        let shard_dir = data_dir.join("shard");
        std::fs::create_dir_all(&shard_dir).context("failed to create shard dir")?;

        let embedders = Embedders::new(models_dir)?;
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
            tuning: Tuning::default(),
        };
        engine.ensure_indexes();
        engine.rebuild_entity_index();
        Ok(engine)
    }

    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    pub fn set_tuning(&mut self, t: Tuning) {
        self.tuning = t;
    }

    /// Rebuild the entity name -> id map from existing Entity nodes.
    fn rebuild_entity_index(&self) {
        if let Ok(all) = self.list() {
            if let Ok(mut idx) = self.entity_index.lock() {
                idx.clear();
                for m in all.iter().filter(|m| m.kind == MemoryKind::Entity) {
                    idx.insert(entities::normalize(&m.title), m.id.clone());
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
        let keyword_fields = [
            "kind", "site_id", "asset_id", "tags", "sync_state", "source", "parent_id",
            "topic_id",
        ];
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
        self.add_opts(input, true)
    }

    /// `add` with control over the disk flush. Batched paths (document
    /// ingestion) pass `flush = false` and flush once at the end: updates are
    /// searchable in memory immediately; the flush only makes them durable.
    fn add_opts(&self, input: NewMemory, flush: bool) -> Result<Memory> {
        let now = chrono::Utc::now().to_rfc3339();
        let id = ulid::Ulid::generate().to_string();
        // An explicit "local only" at creation is the user's choice and binds
        // the policy for this memory from now on.
        if input.sensitivity == Sensitivity::LocalOnly {
            if let Ok(mut m) = self.meta.lock() {
                let _ = m.set_policy_override(&id, false);
            }
        }
        let origin = self.device_id();
        let mut memory = Memory {
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
            sync_state: SyncState::Pending,
            version: 1,
            parent_id: input.parent_id,
            topic_id: input.topic_id,
            archived: false,
            due_at: input.due_at,
            done: false,
            sync_reason: String::new(),
            supersedes: None,
            superseded_by: None,
            origin,
        };
        // The sync policy decides what may leave the device (stamped on the
        // returned value too, so callers see the decision).
        self.apply_policy(&mut memory);
        self.upsert(&memory)?;
        if flush {
            self.shard
                .flush()
                .map_err(|e| anyhow!("flush failed: {e}"))?;
        }
        // Establish semantic relationships (skip chunks: they connect via
        // part_of; skip entities and topics: they are hubs connected via
        // mentions / in_topic).
        if memory.kind != MemoryKind::DocChunk
            && memory.kind != MemoryKind::Entity
            && memory.kind != MemoryKind::Topic
        {
            self.link_related(&memory);
        }
        Ok(memory)
    }

    // ---- entities --------------------------------------------------------

    /// Find an existing entity node by normalized name, or create a typed one.
    /// Dedup key merges aliases ("Stanford" ~ "Stanford University").
    fn find_or_create_entity(&self, name: &str, etype: &str) -> Result<String> {
        let key = entities::normalize(name);
        if key.is_empty() {
            return Err(anyhow!("empty entity name"));
        }
        if let Ok(idx) = self.entity_index.lock() {
            if let Some(id) = idx.get(&key) {
                return Ok(id.clone());
            }
        }
        let mut tags = vec!["entity".to_string()];
        if !etype.is_empty() {
            tags.push(etype.to_string());
        }
        let entity = self.add(NewMemory {
            kind: MemoryKind::Entity,
            title: name.trim().to_string(),
            text: name.trim().to_string(),
            site_id: String::new(),
            asset_id: String::new(),
            geo: None,
            tags,
            source: MemorySource::Manual,
            sensitivity: Sensitivity::Shareable,
            parent_id: None,
            topic_id: None,
            due_at: None,
        })?;
        if let Ok(mut idx) = self.entity_index.lock() {
            idx.insert(key, entity.id.clone());
        }
        Ok(entity.id)
    }

    /// Extract typed entities from `text` and link `from_id` to them via "mentions".
    pub fn attach_entities(&self, from_id: &str, text: &str, model: &str) -> Result<Vec<String>> {
        let ents = entities::extract(&self.ollama, model, text)?;
        let mut ids = Vec::new();
        for e in ents {
            if let Ok(id) = self.find_or_create_entity(&e.name, &e.etype) {
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
        self.upsert_many(std::slice::from_ref(mem))
    }

    /// Batched upsert: one batched embedding call and one shard update for all.
    /// Every write passes through the sync policy (so content edits re-check
    /// what may leave the device).
    fn upsert_many(&self, mems: &[Memory]) -> Result<()> {
        if mems.is_empty() {
            return Ok(());
        }
        let mems: Vec<Memory> = mems
            .iter()
            .map(|m| {
                let mut m = m.clone();
                self.apply_policy(&mut m);
                m
            })
            .collect();
        let texts: Vec<String> = mems.iter().map(|m| m.text.clone()).collect();
        let vectors = self.embedders.embed_documents(&texts)?;
        let mut points: Vec<PointStructPersisted> = Vec::with_capacity(mems.len());
        for (mem, (dense, sparse)) in mems.iter().zip(vectors) {
            let named = Vectors::new_named(vec![
                (DENSE_NAME, Vector::new_dense(dense)),
                (SPARSE_NAME, Vector::from(sparse)),
            ]);
            let payload =
                serde_json::to_value(mem).context("failed to serialize memory payload")?;
            points.push(PointStruct::new(point_id(&mem.id)?, named, payload).into());
        }
        self.shard
            .update(UpdateOperation::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperations::PointsList(points)),
            ))
            .map_err(|e| anyhow!("upsert failed: {e}"))?;
        Ok(())
    }

    // ---- version chains (evolving / conflicting information) ------------

    /// If memory `new_id` updates an OLDER memory about the same thing (a
    /// changed date, number, status, decision or preference), chain them: the
    /// old one is marked superseded, the new one records what it replaced and
    /// takes the next version number, and an "updates" edge is drawn. Only
    /// same-kind memories are compared (notes with notes, events with events).
    /// `model` judges candidates with the local LLM; empty = deterministic
    /// rules only. Returns the id of the memory that was superseded.
    pub fn link_versions(&self, new_id: &str, model: &str) -> Result<Option<String>> {
        let Some(new) = self.get(new_id)? else {
            return Ok(None);
        };
        if !is_versionable(&new.kind) || new.supersedes.is_some() {
            return Ok(None);
        }
        let (dense, _) = self.embedders.embed_query(&new.text)?;
        // Ensemble: the precise rules decide first; only when they say no does
        // the LLM judge the closest candidates (up to 2, bounded cost), which
        // catches revisions that need world knowledge ("I live in Pune" ->
        // "I moved to Bangalore").
        let mut judged = 0;
        for c in self.query_points(&dense, None, Some(kind_filter(new.kind.clone())), 6)? {
            let old = c.memory;
            // Older (ULIDs sort by time), current, and plausibly about the same thing.
            if old.id >= new.id || old.superseded_by.is_some() || c.score < VERSION_MIN_SIM {
                continue;
            }
            let mut updates = heuristic_updates(&old, &new, c.score);
            if !updates && !model.trim().is_empty() && judged < 2 {
                judged += 1;
                updates = self.judge_update(&old, &new, model).unwrap_or(false);
            }
            if updates {
                self.chain_versions(&old, &new)?;
                return Ok(Some(old.id));
            }
        }
        Ok(None)
    }

    /// Ask the local LLM whether NEW updates OLD.
    fn judge_update(&self, old: &Memory, new: &Memory, model: &str) -> Result<bool> {
        let system = "You compare two memories from a personal memory system. OLD was \
recorded before NEW. Return ONLY JSON {\"relation\": \"updates\" | \"same\" | \"different\"}. \
\"updates\": NEW changes, corrects, or replaces a fact in OLD about the SAME thing (a changed \
date, time, number, status, place, decision, or preference), so OLD is no longer current. \
\"same\": NEW only restates OLD. \"different\": they are about different things, or both can \
be true at once.";
        let prompt = format!("OLD: {}\nNEW: {}", truncate(&old.text, 400), truncate(&new.text, 400));
        let raw = self.ollama.generate(model, Some(system), &prompt, true)?;
        let v = crate::chat::parse_json_lenient(&raw).ok_or_else(|| anyhow!("bad judge JSON"))?;
        let rel = v["relation"].as_str().unwrap_or_default().to_lowercase();
        Ok(rel.contains("update"))
    }

    /// Record `new` as the next version of `old`. Marking the old memory is an
    /// edit (updated_at moves), so the change propagates on the next sync.
    fn chain_versions(&self, old: &Memory, new: &Memory) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.set_payload_fields(
            &old.id,
            serde_json::json!({ "superseded_by": new.id, "updated_at": now }),
        )?;
        self.set_payload_fields(
            &new.id,
            serde_json::json!({ "supersedes": old.id, "version": old.version + 1 }),
        )?;
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.add_edge(&new.id, &old.id, "updates");
        }
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(())
    }

    /// Follow `superseded_by` links to the current version (bounded).
    fn latest_version(&self, m: Memory) -> Memory {
        let mut cur = m;
        for _ in 0..8 {
            let Some(next) = cur.superseded_by.clone() else { break };
            match self.get(&next) {
                Ok(Some(n)) => cur = n,
                _ => break,
            }
        }
        cur
    }

    /// The version chain containing `id`, oldest first (for the UI).
    pub fn version_chain(&self, id: &str) -> Result<Vec<Memory>> {
        let Some(start) = self.get(id)? else {
            return Ok(Vec::new());
        };
        let mut chain = vec![start];
        for _ in 0..8 {
            let Some(prev) = chain[0].supersedes.clone() else { break };
            match self.get(&prev)? {
                Some(p) => chain.insert(0, p),
                None => break,
            }
        }
        for _ in 0..8 {
            let Some(next) = chain.last().and_then(|m| m.superseded_by.clone()) else { break };
            match self.get(&next)? {
                Some(n) => chain.push(n),
                None => break,
            }
        }
        Ok(chain)
    }

    // ---- sync policy (what may leave the device) ------------------------

    /// This device's stable id.
    pub fn device_id(&self) -> String {
        self.meta.lock().map(|mut m| m.device_id()).unwrap_or_default()
    }

    /// The sync decision for a memory. Order: the user's explicit choice;
    /// derived data (entity hubs) stays local; sensitive content stays local
    /// (see `policy::scan`); everything else may sync.
    pub fn sync_decision(&self, m: &Memory) -> SyncDecision {
        let over = self.meta.lock().ok().and_then(|x| x.policy_override(&m.id));
        if let Some(share) = over {
            return if share {
                SyncDecision::share("user", "Syncs: you allowed it")
            } else {
                SyncDecision::local("user", "Stays on device: you chose to keep it local")
            };
        }
        if m.kind == MemoryKind::Entity {
            return SyncDecision::local("derived", "Stays on device: derived link, rebuilt on each device");
        }
        // A topic is only as shareable as its members: if every member is
        // private, the topic (whose name/summary derive from them) is too.
        if m.kind == MemoryKind::Topic {
            if let Ok(members) = self.topic_members(&m.id) {
                if !members.is_empty() && members.iter().all(|x| !self.member_shareable(x)) {
                    return SyncDecision::local("derived", "Stays on device: all of its memories are private");
                }
            }
        }
        if let Some(f) = policy::scan(&format!("{}\n{}", m.title, m.text)) {
            return SyncDecision::local(f.category, &f.reason);
        }
        match m.kind {
            MemoryKind::Topic | MemoryKind::Summary => {
                SyncDecision::share("knowledge", "Syncs: consolidated knowledge")
            }
            MemoryKind::Document | MemoryKind::DocChunk => {
                SyncDecision::share("document", "Syncs: document knowledge")
            }
            _ => SyncDecision::share("memory", "Syncs: memory"),
        }
    }

    /// May this (non-topic) memory's content feed shared artifacts such as
    /// topic summaries? Same rules as `sync_decision`, without topic recursion.
    fn member_shareable(&self, m: &Memory) -> bool {
        if let Some(share) = self.meta.lock().ok().and_then(|x| x.policy_override(&m.id)) {
            return share;
        }
        m.kind != MemoryKind::Entity && policy::scan(&format!("{}\n{}", m.title, m.text)).is_none()
    }

    /// Stamp the policy decision onto a memory before it is written.
    fn apply_policy(&self, m: &mut Memory) {
        let d = self.sync_decision(m);
        m.sync_reason = d.reason;
        if d.share {
            m.sensitivity = Sensitivity::Shareable;
            if m.sync_state == SyncState::LocalOnly {
                m.sync_state = SyncState::Pending;
            }
        } else {
            m.sensitivity = Sensitivity::LocalOnly;
            m.sync_state = SyncState::LocalOnly;
        }
    }

    /// User override: allow a memory to sync, or keep it on this device.
    /// Re-stamps the memory (and a document's chunks) payload-only.
    pub fn set_sync_override(&self, id: &str, share: bool) -> Result<SyncDecision> {
        let m = self.get(id)?.ok_or_else(|| anyhow!("memory not found"))?;
        if let Ok(mut meta) = self.meta.lock() {
            meta.set_policy_override(id, share)?;
        }
        let mut targets = vec![m];
        if targets[0].kind == MemoryKind::Document {
            targets.extend(self.scroll_where(Filter {
                must: Some(vec![match_cond("parent_id", id)]),
                ..Default::default()
            })?);
            if let Ok(mut meta) = self.meta.lock() {
                for c in targets.iter().skip(1) {
                    meta.set_policy_override(&c.id, share)?;
                }
            }
        }
        for t in &targets {
            let mut t = t.clone();
            self.apply_policy(&mut t);
            // Changing the decision is an edit: bump updated_at so sync acts.
            let now = chrono::Utc::now().to_rfc3339();
            self.set_payload_fields(
                &t.id,
                serde_json::json!({
                    "sensitivity": t.sensitivity, "sync_state": t.sync_state,
                    "sync_reason": t.sync_reason, "updated_at": now,
                }),
            )?;
        }
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        let m = self.get(id)?.ok_or_else(|| anyhow!("memory not found"))?;
        Ok(self.sync_decision(&m))
    }

    /// What stays on the device and why, and what may sync (for the UI).
    pub fn policy_summary(&self) -> Result<PolicySummary> {
        let mut s = PolicySummary::default();
        for m in self.list()? {
            if m.kind == MemoryKind::Entity {
                continue;
            }
            let d = self.sync_decision(&m);
            if d.share {
                s.shared += 1;
            } else {
                s.local += 1;
                *s.local_by_category.entry(d.category.clone()).or_insert(0) += 1;
                if s.recent_local.len() < 8 {
                    s.recent_local.push(PolicyItem {
                        id: m.id.clone(),
                        title: if m.title.trim().is_empty() {
                            truncate(&m.text, 60)
                        } else {
                            m.title.clone()
                        },
                        category: d.category,
                        reason: d.reason,
                    });
                }
            }
        }
        Ok(s)
    }

    /// Merge payload fields on a point WITHOUT re-embedding (metadata updates).
    fn set_payload_fields(&self, id: &str, fields: serde_json::Value) -> Result<()> {
        let pid = point_id(id)?;
        let payload: Payload =
            serde_json::from_value(fields).context("invalid payload fields")?;
        self.shard
            .update(UpdateOperation::PayloadOperation(PayloadOps::SetPayload(
                SetPayloadOp {
                    payload,
                    points: Some(vec![pid]),
                    filter: None,
                    key: None,
                },
            )))
            .map_err(|e| anyhow!("set payload failed: {e}"))?;
        Ok(())
    }

    /// Record that these memories were retrieved (feeds salience).
    pub fn bump_access(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        if let Ok(mut meta) = self.meta.lock() {
            let _ = meta.bump_access(ids);
        }
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
        // Record a tombstone so the deletion propagates on the next sync.
        if let Ok(mut meta) = self.meta.lock() {
            let _ = meta.add_tombstone(id, &chrono::Utc::now().to_rfc3339());
        }
        Ok(())
    }

    // ---- topics (the schema / organizing layer) -------------------------

    /// Route a memory to a topic and persist the assignment (payload-only, no
    /// re-embed). Returns the topic id it was filed under, or None on failure.
    /// `text` grounds routing; `model` (may be empty) names a new topic.
    pub fn route_and_assign(&self, mem_id: &str, text: &str, model: &str) -> Option<String> {
        self.route_and_assign_opts(mem_id, text, model, true, None)
    }

    /// `name_hint` names a newly created topic without an LLM call (documents
    /// pass their title-derived name).
    fn route_and_assign_opts(
        &self,
        mem_id: &str,
        text: &str,
        model: &str,
        flush: bool,
        name_hint: Option<&str>,
    ) -> Option<String> {
        let topic_id = self.route_to_topic(text, model, mem_id, name_hint).ok().flatten()?;
        self.assign_topic(mem_id, &topic_id, flush);
        Some(topic_id)
    }

    /// File a memory under a topic (payload-only), touch the topic, and record
    /// the in_topic edge. Shared by routing and the user's manual overrides.
    fn assign_topic(&self, mem_id: &str, topic_id: &str, flush: bool) {
        let _ = self.set_payload_fields(mem_id, serde_json::json!({ "topic_id": topic_id }));
        let _ = self.set_payload_fields(
            topic_id,
            serde_json::json!({ "updated_at": chrono::Utc::now().to_rfc3339() }),
        );
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.add_in_topic(mem_id, topic_id);
        }
        if let Ok(mut m) = self.meta.lock() {
            let _ = m.mark_topic_changed(topic_id);
        }
        if flush {
            let _ = self.shard.flush();
        }
    }

    /// Route by k-nearest-neighbour vote: memories that are already filed and
    /// similar to `text` vote for their topic (weighted by similarity), plus the
    /// topic nodes themselves (name + rolling summary). This compares new content
    /// with a topic's actual CONTENT, not just its 2-4 word name. If nothing is
    /// close enough, a new topic is minted and named. Never flushes: callers own
    /// durability. `exclude` is the memory being routed (and its chunks).
    fn route_to_topic(
        &self,
        text: &str,
        model: &str,
        exclude: &str,
        name_hint: Option<&str>,
    ) -> Result<Option<String>> {
        let t = self.tuning;
        let probe: String = text.chars().take(2000).collect();
        if probe.trim().is_empty() {
            return Ok(None);
        }
        let (dense, _) = self.embedders.embed_query(&probe)?;

        let mut votes: HashMap<String, f32> = HashMap::new();
        for h in self.query_points(&dense, None, Some(content_filter()), 12)? {
            if h.memory.id == exclude || h.memory.parent_id.as_deref() == Some(exclude) {
                continue;
            }
            if h.score < t.topic_join {
                continue;
            }
            if let Some(tid) = h.memory.topic_id {
                *votes.entry(tid).or_insert(0.0) += h.score;
            }
        }
        for h in self.query_points(&dense, None, Some(kind_filter(MemoryKind::Topic)), 3)? {
            if h.score >= t.topic_join_direct {
                *votes.entry(h.memory.id).or_insert(0.0) += h.score;
            }
        }
        // Graph evidence: memories that mention the same specific entities
        // (e.g. "I-20" and "DSO") belong together even when their wording is
        // too different for embeddings of short text to agree.
        for (tid, score) in self.entity_votes(exclude) {
            *votes.entry(tid).or_insert(0.0) += score;
        }
        if let Some((tid, _)) = votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            return Ok(Some(tid));
        }

        // Nothing close: create a topic, named by the hint (e.g. the document's
        // title) or else from the content.
        // Never derive a name from private content (topic names can sync).
        let name = if policy::scan(&probe).is_some() {
            "Private".to_string()
        } else {
            match name_hint.map(str::trim).filter(|h| !h.is_empty()) {
                Some(h) => h.to_string(),
                None => self.name_topic(&probe, model),
            }
        };
        Ok(Some(self.create_topic(&name, &[])?))
    }

    /// Topic votes from shared entities: for each topic, ENTITY_VOTE per
    /// distinct specific entity that `mem_id` shares with that topic's members.
    /// Only topics backed by 2+ shared entities vote (one incidental shared
    /// word is not enough), and generic entities mentioned by many memories are
    /// ignored.
    fn entity_votes(&self, mem_id: &str) -> Vec<(String, f32)> {
        const ENTITY_VOTE: f32 = 0.3;
        const GENERIC_DF: usize = 10;
        let (mine, holders): (Vec<String>, HashMap<String, Vec<String>>) = {
            let Ok(g) = self.graph.lock() else {
                return Vec::new();
            };
            let mine: Vec<String> = g
                .all()
                .iter()
                .filter(|e| e.relation == "mentions" && e.from == mem_id)
                .map(|e| e.to.clone())
                .collect();
            let mut holders: HashMap<String, Vec<String>> = HashMap::new();
            for e in g.all() {
                if e.relation == "mentions" && e.from != mem_id && mine.contains(&e.to) {
                    holders.entry(e.to.clone()).or_default().push(e.from.clone());
                }
            }
            (mine, holders)
        };
        if mine.is_empty() {
            return Vec::new();
        }
        // topic -> set of shared entities
        let mut shared: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
        for (ent, mems) in holders {
            if mems.len() > GENERIC_DF {
                continue;
            }
            for m in mems {
                if let Ok(Some(mem)) = self.get(&m) {
                    if let Some(tid) = mem.topic_id {
                        shared.entry(tid).or_default().insert(ent.clone());
                    }
                }
            }
        }
        shared
            .into_iter()
            .filter(|(_, ents)| ents.len() >= 2)
            .map(|(tid, ents)| (tid, ENTITY_VOTE * ents.len() as f32))
            .collect()
    }

    /// Create an (unsummarized) topic node and return its id.
    fn create_topic(&self, name: &str, extra_tags: &[&str]) -> Result<String> {
        let mut tags = vec!["topic".to_string()];
        tags.extend(extra_tags.iter().map(|s| s.to_string()));
        let topic = self.add_opts(
            NewMemory {
                kind: MemoryKind::Topic,
                title: name.to_string(),
                text: name.to_string(),
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags,
                source: MemorySource::Manual,
                sensitivity: Sensitivity::Shareable,
                parent_id: None,
                topic_id: None,
                due_at: None,
            },
            false,
        )?;
        Ok(topic.id)
    }

    /// The dedicated topic for identity documents (resume / CV / about-me), so
    /// the user's profile never dissolves into whichever research topic it
    /// happens to mention.
    fn profile_topic(&self) -> Result<String> {
        if let Some(t) = self
            .list_kind(MemoryKind::Topic)?
            .into_iter()
            .find(|t| t.tags.iter().any(|x| x == "profile"))
        {
            return Ok(t.id);
        }
        self.create_topic("Profile", &["profile"])
    }

    // ---- topic management + rolling summaries ---------------------------

    /// Content members of a topic (documents, notes, events...; not chunks).
    pub fn topic_members(&self, topic_id: &str) -> Result<Vec<Memory>> {
        let mut f = content_filter();
        f.must = Some(vec![match_cond("topic_id", topic_id)]);
        if let Some(nots) = f.must_not.as_mut() {
            nots.push(match_cond("kind", "doc_chunk"));
        }
        self.scroll_where(f)
    }

    fn topic_member_count(&self, topic_id: &str) -> usize {
        let mut f = content_filter();
        f.must = Some(vec![match_cond("topic_id", topic_id)]);
        if let Some(nots) = f.must_not.as_mut() {
            nots.push(match_cond("kind", "doc_chunk"));
        }
        self.shard
            .count(CountRequestBuilder::new().filter(f).exact(true).build())
            .unwrap_or(0)
    }

    /// All non-empty topics with member counts, largest first.
    pub fn list_topics(&self) -> Result<Vec<TopicInfo>> {
        let mut out: Vec<TopicInfo> = Vec::new();
        for t in self.list_kind(MemoryKind::Topic)? {
            let members = self.topic_member_count(&t.id);
            if members == 0 {
                continue;
            }
            out.push(self.topic_info(&t, members));
        }
        out.sort_by(|a, b| b.members.cmp(&a.members).then_with(|| a.name.cmp(&b.name)));
        Ok(out)
    }

    fn topic_info(&self, t: &Memory, members: usize) -> TopicInfo {
        let user_named = self
            .meta
            .lock()
            .map(|m| m.topic_state(&t.id).user_named)
            .unwrap_or(false);
        TopicInfo {
            id: t.id.clone(),
            name: t.title.clone(),
            summary: topic_summary(t),
            members,
            user_named,
            updated_at: t.updated_at.clone(),
        }
    }

    /// Distill a topic's members into a rolling summary (and a better name,
    /// unless the user named it). The summary becomes the topic node's text and
    /// is EMBEDDED, so routing and query-time activation match what the subject
    /// is actually about. Local LLM (Ollama).
    pub fn refresh_topic(&self, topic_id: &str, model: &str) -> Result<TopicInfo> {
        let mut topic = self
            .get(topic_id)?
            .filter(|m| m.kind == MemoryKind::Topic)
            .ok_or_else(|| anyhow!("topic not found"))?;
        let members = self.topic_members(topic_id)?;
        if members.is_empty() {
            return Err(anyhow!("topic has no members"));
        }
        // Only members that may sync feed the summary: the topic node syncs, so
        // private content must never leak into it.
        let members: Vec<Memory> = members.into_iter().filter(|m| self.member_shareable(m)).collect();
        if members.is_empty() {
            return Err(anyhow!("topic has no shareable members to summarize"));
        }
        let mut notes = String::new();
        for m in &members {
            let kind = kind_str(&m.kind);
            let line = if m.kind == MemoryKind::Document {
                format!("- [{kind}] {}: {}\n", m.title, truncate(&m.text, 320))
            } else {
                format!("- [{kind}] {}\n", truncate(&m.text, 320))
            };
            if notes.chars().count() + line.chars().count() > 3600 {
                break;
            }
            notes.push_str(&line);
        }
        let system = "You maintain a personal knowledge base. Given the memories filed \
under one subject, return ONLY JSON: {\"name\": \"a 2 to 4 word Title Case name for the \
subject\", \"summary\": \"3 to 5 factual sentences on what the user knows, decided, and is \
working on in this subject\"}. Keep the current name unless it is clearly inaccurate. Use \
only the memories provided; do not invent facts.";
        let prompt = format!("Current name: {}\nMemories:\n{notes}", topic.title);
        let raw = self.ollama.generate(model, Some(system), &prompt, true)?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("bad summary JSON: {e}"))?;
        let summary = v["summary"].as_str().unwrap_or_default().trim().to_string();
        if summary.is_empty() {
            return Err(anyhow!("model returned an empty summary"));
        }
        let user_named = self
            .meta
            .lock()
            .map(|m| m.topic_state(topic_id).user_named)
            .unwrap_or(false);
        if !user_named {
            let name = clean_label(v["name"].as_str().unwrap_or_default());
            if !name.is_empty() {
                topic.title = name;
            }
        }
        topic.text = truncate(&summary, 1200);
        topic.updated_at = chrono::Utc::now().to_rfc3339();
        topic.sync_state = SyncState::Pending; // policy re-checks on write
        self.upsert(&topic)?;
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        if let Ok(mut m) = self.meta.lock() {
            let _ = m.mark_topic_summarized(topic_id, &topic.updated_at);
        }
        Ok(self.topic_info(&topic, members.len()))
    }

    /// Refresh one topic's summary if it was never summarized or has gathered
    /// at least `min_changes` new members since. Returns whether it refreshed.
    pub fn refresh_topic_if_due(&self, topic_id: &str, model: &str, min_changes: u32) -> bool {
        if model.trim().is_empty() {
            return false;
        }
        let st = self
            .meta
            .lock()
            .map(|m| m.topic_state(topic_id))
            .unwrap_or_default();
        let due = st.summarized_at.is_none() || st.changes_since_summary >= min_changes;
        due && self.refresh_topic(topic_id, model).is_ok()
    }

    /// Refresh every due topic (most-changed first), at most `max`.
    pub fn refresh_due_topics(&self, model: &str, min_changes: u32, max: usize) -> usize {
        let Ok(topics) = self.list_kind(MemoryKind::Topic) else {
            return 0;
        };
        let mut due: Vec<(String, u32)> = topics
            .iter()
            .filter_map(|t| {
                let st = self.meta.lock().ok()?.topic_state(&t.id);
                let is_due = st.summarized_at.is_none() || st.changes_since_summary >= min_changes;
                is_due.then_some((t.id.clone(), st.changes_since_summary))
            })
            .collect();
        due.sort_by(|a, b| b.1.cmp(&a.1));
        due.into_iter()
            .take(max)
            .filter(|(id, _)| self.topic_member_count(id) > 0 && self.refresh_topic(id, model).is_ok())
            .count()
    }

    /// User override: rename a topic. Automatic summaries keep this name.
    pub fn rename_topic(&self, topic_id: &str, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(anyhow!("name cannot be empty"));
        }
        let mut t = self
            .get(topic_id)?
            .filter(|m| m.kind == MemoryKind::Topic)
            .ok_or_else(|| anyhow!("topic not found"))?;
        let unsummarized = topic_summary(&t).is_empty();
        t.title = name.to_string();
        if unsummarized {
            t.text = name.to_string();
        }
        t.updated_at = chrono::Utc::now().to_rfc3339();
        t.sync_state = SyncState::Pending; // policy re-checks on write
        self.upsert(&t)?;
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        if let Ok(mut m) = self.meta.lock() {
            let _ = m.set_topic_user_named(topic_id);
        }
        Ok(())
    }

    /// User override: fold every member of `from` into `into`, then remove
    /// `from`. Payload-only (no re-embedding).
    pub fn merge_topics(&self, from: &str, into: &str) -> Result<()> {
        if from == into {
            return Err(anyhow!("cannot merge a topic into itself"));
        }
        for id in [from, into] {
            self.get(id)?
                .filter(|m| m.kind == MemoryKind::Topic)
                .ok_or_else(|| anyhow!("topic not found"))?;
        }
        self.set_payload_where(
            Filter {
                must: Some(vec![match_cond("topic_id", from)]),
                ..Default::default()
            },
            serde_json::json!({ "topic_id": into }),
        )?;
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.retarget_topic(from, into);
        }
        self.delete(from)?;
        if let Ok(mut m) = self.meta.lock() {
            let _ = m.remove_topic(from);
            let _ = m.mark_topic_changed(into);
        }
        Ok(())
    }

    /// User override: file a memory (and a document's chunks) under a topic.
    pub fn move_to_topic(&self, mem_id: &str, topic_id: &str) -> Result<()> {
        self.get(topic_id)?
            .filter(|m| m.kind == MemoryKind::Topic)
            .ok_or_else(|| anyhow!("topic not found"))?;
        let mem = self.get(mem_id)?.ok_or_else(|| anyhow!("memory not found"))?;
        self.assign_topic(mem_id, topic_id, false);
        if mem.kind == MemoryKind::Document {
            self.set_payload_where(
                Filter {
                    must: Some(vec![match_cond("parent_id", mem_id)]),
                    ..Default::default()
                },
                serde_json::json!({ "topic_id": topic_id }),
            )?;
        }
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(())
    }

    /// Organize existing memories: route every unfiled document/note into a
    /// topic (oldest first, so subjects form in the order they were learned),
    /// propagate to document chunks, summarize due topics, and drop empties.
    pub fn organize(&self, model: &str) -> Result<OrganizeReport> {
        let mut report = OrganizeReport::default();
        let mut pending: Vec<Memory> = self
            .list()?
            .into_iter()
            .filter(|m| {
                m.topic_id.is_none()
                    && !m.archived
                    && !matches!(
                        m.kind,
                        MemoryKind::DocChunk | MemoryKind::Topic | MemoryKind::Entity | MemoryKind::Summary
                    )
            })
            .collect();
        pending.sort_by(|a, b| a.id.cmp(&b.id));
        for m in &pending {
            let topic = if m.kind == MemoryKind::Document {
                let probe = self.document_probe(m);
                let tid = if is_profile_title(&m.title) {
                    self.profile_topic().ok().map(|t| {
                        self.assign_topic(&m.id, &t, false);
                        t
                    })
                } else {
                    let hint = name_from_title(&m.title);
                    self.route_and_assign_opts(&m.id, &probe, "", false, Some(&hint))
                };
                if let Some(t) = &tid {
                    let _ = self.set_payload_where(
                        Filter {
                            must: Some(vec![match_cond("parent_id", &m.id)]),
                            ..Default::default()
                        },
                        serde_json::json!({ "topic_id": t }),
                    );
                }
                tid
            } else {
                self.route_and_assign_opts(&m.id, &m.text, "", false, None)
            };
            if topic.is_some() {
                report.routed += 1;
            }
        }
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        let _ = self.consolidate_topics();
        if !model.trim().is_empty() {
            report.summarized = self.refresh_due_topics(model, 1, 16);
        }
        let _ = self.prune_empty_topics();
        report.topics = self.list_topics()?.len();
        Ok(report)
    }

    /// Text that represents a stored document for routing: its first chunks.
    fn document_probe(&self, doc: &Memory) -> String {
        let mut chunks = self
            .scroll_where(Filter {
                must: Some(vec![match_cond("parent_id", &doc.id)]),
                ..Default::default()
            })
            .unwrap_or_default();
        chunks.sort_by_key(|c| chunk_index(&c.title));
        let mut probe = doc.text.clone();
        for c in chunks.iter().take(2) {
            probe.push('\n');
            probe.push_str(&c.text);
        }
        probe
    }

    /// Topic consolidation (re-homing). Online routing is order-dependent: the
    /// first note of a subject can arrive before anything similar is filed and
    /// found its own topic. As knowledge grows, members of small topics (<= 2)
    /// are re-examined and moved to an established topic when their filed
    /// neighbours there now clearly claim them (two neighbours, or one very
    /// strong one, and more support than their current topic). Note-only
    /// singletons are processed first so they move toward document-anchored
    /// subjects. User-named topics and the Profile topic are never touched.
    /// Deterministic, no LLM. Returns how many memories moved.
    pub fn consolidate_topics(&self) -> Result<usize> {
        let t = self.tuning;
        let mut small: Vec<(String, usize, bool)> = Vec::new();
        for topic in self.list_kind(MemoryKind::Topic)? {
            if topic.tags.iter().any(|x| x == "profile") {
                continue;
            }
            let user_named = self
                .meta
                .lock()
                .map(|m| m.topic_state(&topic.id).user_named)
                .unwrap_or(false);
            if user_named {
                continue;
            }
            let members = self.topic_members(&topic.id)?;
            if members.is_empty() || members.len() > 2 {
                continue;
            }
            let has_doc = members.iter().any(|m| m.kind == MemoryKind::Document);
            small.push((topic.id, members.len(), has_doc));
        }
        small.sort_by_key(|(_, n, has_doc)| (*n, *has_doc));

        let mut moved = 0;
        for (tid, _, _) in small {
            for m in self.topic_members(&tid)? {
                let probe = if m.kind == MemoryKind::Document {
                    self.document_probe(&m)
                } else {
                    m.text.clone()
                };
                let (dense, _) = self.embedders.embed_query(&truncate(&probe, 2000))?;
                let mut votes: HashMap<String, (f32, f32, usize)> = HashMap::new();
                for h in self.query_points(&dense, None, Some(content_filter()), 12)? {
                    if h.memory.id == m.id || h.memory.parent_id.as_deref() == Some(&m.id) {
                        continue;
                    }
                    if h.score < t.topic_join {
                        continue;
                    }
                    if let Some(other) = h.memory.topic_id {
                        let e = votes.entry(other).or_insert((0.0, 0.0, 0));
                        e.0 += h.score;
                        e.1 = e.1.max(h.score);
                        e.2 += 1;
                    }
                }
                // Summarized topics vote as themselves: a rolling summary is
                // strong evidence of what the subject covers.
                for h in self.query_points(&dense, None, Some(kind_filter(MemoryKind::Topic)), 3)? {
                    if h.memory.id != tid
                        && !topic_summary(&h.memory).is_empty()
                        && h.score >= t.topic_join_direct
                    {
                        let e = votes.entry(h.memory.id).or_insert((0.0, 0.0, 0));
                        e.0 += h.score;
                        e.1 = e.1.max(h.score);
                        e.2 += 1;
                    }
                }
                for (other, score) in self.entity_votes(&m.id) {
                    let e = votes.entry(other).or_insert((0.0, 0.0, 0));
                    e.0 += score;
                    e.1 = e.1.max(score);
                    e.2 += 2; // backed by 2+ shared specific entities
                }
                let own = votes.get(&tid).map(|v| v.0).unwrap_or(0.0);
                let best = votes
                    .iter()
                    .filter(|(k, _)| k.as_str() != tid)
                    .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap_or(std::cmp::Ordering::Equal));
                if let Some((dest, (sum, max, n))) = best {
                    let claimed = *n >= 2 || *max >= t.topic_join + 0.15;
                    if claimed && *sum > own {
                        let dest = dest.clone();
                        self.move_to_topic(&m.id, &dest)?;
                        moved += 1;
                    }
                }
            }
        }
        if moved > 0 {
            let _ = self.prune_empty_topics();
        }
        Ok(moved)
    }

    /// Remove topics that no longer have members (after merges/moves).
    /// Only topics created here and never synced are pruned: a synced topic
    /// may have members on other devices, and deleting it would propagate.
    pub fn prune_empty_topics(&self) -> Result<usize> {
        let device = self.device_id();
        let mut n = 0;
        for t in self.list_kind(MemoryKind::Topic)? {
            let foreign = !t.origin.is_empty() && t.origin != device;
            let ever_synced = self
                .meta
                .lock()
                .map(|m| m.synced_version(&t.id).is_some())
                .unwrap_or(true);
            if foreign || ever_synced {
                continue;
            }
            if self.topic_member_count(&t.id) == 0 {
                self.delete(&t.id)?;
                if let Ok(mut m) = self.meta.lock() {
                    let _ = m.remove_topic(&t.id);
                }
                n += 1;
            }
        }
        Ok(n)
    }

    /// Payload-only update of every point matching `filter`.
    fn set_payload_where(&self, filter: Filter, fields: serde_json::Value) -> Result<()> {
        let payload: Payload =
            serde_json::from_value(fields).context("invalid payload fields")?;
        self.shard
            .update(UpdateOperation::PayloadOperation(PayloadOps::SetPayload(
                SetPayloadOp {
                    payload,
                    points: None,
                    filter: Some(filter),
                    key: None,
                },
            )))
            .map_err(|e| anyhow!("set payload failed: {e}"))?;
        Ok(())
    }

    /// Generate a short topic label from content (fast model; heuristic fallback).
    fn name_topic(&self, text: &str, model: &str) -> String {
        if !model.trim().is_empty() {
            let system = "You name the subject of a note or document for a personal knowledge \
base. Reply with ONLY a 2 to 4 word topic label in Title Case. No quotes, no punctuation, \
no explanation.";
            if let Ok(s) = self.ollama.generate(model, Some(system), text, false) {
                let label = clean_label(&s);
                if !label.is_empty() {
                    return label;
                }
            }
        }
        heuristic_label(text)
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
        self.ingest_text(filename, &text, entity_model)
    }

    /// Chunk + store already-extracted text as a Document node with chunk
    /// children. Shared by file ingestion and programmatic clients (API, CLI,
    /// benchmarks).
    pub fn ingest_text(
        &self,
        filename: &str,
        text: &str,
        entity_model: &str,
    ) -> Result<(Memory, usize)> {
        if text.trim().is_empty() {
            return Err(anyhow!("no extractable text in '{filename}'"));
        }
        let preview: String = text.chars().take(400).collect();
        // Everything below is written without intermediate flushes; one flush
        // after the chunk batch makes the whole document durable at once.
        let doc = self.add_opts(
            NewMemory {
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
                topic_id: None,
                due_at: None,
            },
            false,
        )?;

        // Route the document to a topic (schema layer), then file its chunks
        // under the same topic so a research effort stays coherent. Identity
        // documents always go to the dedicated Profile topic.
        let doc_topic = if is_profile_title(filename) {
            self.profile_topic().ok().map(|tid| {
                self.assign_topic(&doc.id, &tid, false);
                tid
            })
        } else {
            let hint = name_from_title(filename);
            self.route_and_assign_opts(&doc.id, text, entity_model, false, Some(&hint))
        };

        // Chunks are stored in ONE batch: a single batched embedding call, one
        // upsert, one flush, and one edge-file write (instead of per chunk).
        let now = chrono::Utc::now().to_rfc3339();
        let chunk_mems: Vec<Memory> = documents::chunk_text(text)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| Memory {
                id: ulid::Ulid::generate().to_string(),
                kind: MemoryKind::DocChunk,
                title: format!("{filename} [{}]", i + 1),
                text: chunk,
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags: Vec::new(),
                source: MemorySource::File,
                captured_at: now.clone(),
                created_at: now.clone(),
                updated_at: now.clone(),
                salience: 0.0,
                sensitivity: Sensitivity::Shareable,
                sync_state: SyncState::Pending,
                version: 1,
                parent_id: Some(doc.id.clone()),
                topic_id: doc_topic.clone(),
                archived: false,
                due_at: None,
                done: false,
                sync_reason: String::new(),
                supersedes: None,
                superseded_by: None,
                origin: doc.origin.clone(),
            })
            .collect();
        let n = chunk_mems.len();
        self.upsert_many(&chunk_mems)?;
        self.shard
            .flush()
            .map_err(|e| anyhow!("flush failed: {e}"))?;
        if let Ok(mut g) = self.graph.lock() {
            let ids: Vec<String> = chunk_mems.iter().map(|m| m.id.clone()).collect();
            let _ = g.add_part_of_many(&ids, &doc.id);
        }
        // Link the document to the entities it mentions (best-effort).
        if !entity_model.is_empty() {
            let _ = self.attach_entities(&doc.id, text, entity_model);
        }
        Ok((doc, n))
    }

    pub fn list_documents(&self) -> Result<Vec<Memory>> {
        self.list_kind(MemoryKind::Document)
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

    /// Server points for many memories with ONE batched embedding call.
    fn to_server_points(&self, mems: &[Memory]) -> Result<Vec<ServerPoint>> {
        let texts: Vec<String> = mems.iter().map(|m| m.text.clone()).collect();
        let vectors = self.embedders.embed_documents(&texts)?;
        mems.iter()
            .zip(vectors)
            .map(|(m, (dense, sparse))| {
                Ok(ServerPoint {
                    id: uuid_string(&m.id)?,
                    dense,
                    sparse_indices: sparse.indices,
                    sparse_values: sparse.values,
                    payload: serde_json::to_value(m)?,
                })
            })
            .collect()
    }

    /// Tiered pull: which remote memories should come down to this device.
    /// Consolidated knowledge (topics, summaries, documents), scheduled events,
    /// and this device's own memories always do; other devices' raw memories
    /// stay in the cloud (searchable via cloud search when online) unless
    /// `pull_all` mirrors everything.
    fn pull_eligible(r: &Memory, device: &str, pull_all: bool) -> bool {
        pull_all
            || r.origin.is_empty()
            || r.origin == device
            || matches!(
                r.kind,
                MemoryKind::Topic
                    | MemoryKind::Summary
                    | MemoryKind::Document
                    | MemoryKind::DocChunk
                    | MemoryKind::Event
            )
    }

    /// Keep the losing side of a real conflict as an older version of the
    /// winner (never drop an edit). Returns the preserved copy.
    fn preserve_conflict_version(&self, loser: &Memory, winner_id: &str) -> Result<Memory> {
        let mut copy = loser.clone();
        copy.id = ulid::Ulid::generate().to_string();
        copy.superseded_by = Some(winner_id.to_string());
        copy.supersedes = None;
        if !copy.tags.iter().any(|t| t == "sync-conflict") {
            copy.tags.push("sync-conflict".to_string());
        }
        copy.sync_state = SyncState::Pending;
        copy.updated_at = chrono::Utc::now().to_rfc3339();
        self.upsert(&copy)?;
        if let Ok(mut g) = self.graph.lock() {
            let _ = g.add_edge(winner_id, &copy.id, "updates");
        }
        Ok(copy)
    }

    /// Two-way edge <-> cloud sync with a Qdrant Server:
    ///   1. deletions (tombstones) propagate both ways;
    ///   2. the sync policy is enforced on every memory (even ones stored before
    ///      the policy existed); cloud copies of now-private memories are
    ///      retracted;
    ///   3. 3-way merge against the last-synced version: if only one side
    ///      changed, it wins cleanly; if BOTH changed it is a real conflict: the
    ///      newer edit wins and the other is preserved as an older version;
    ///   4. tiered pull (see `pull_eligible`), with graph edges rebuilt for
    ///      pulled memories.
    pub fn sync(&self, client: &SyncClient, pull_all: bool) -> Result<SyncReport> {
        client.ensure_collection(DENSE_DIM)?;
        client.ensure_tombstone_collection()?;

        let mut report = SyncReport::default();
        let device = self.device_id();

        // 1. Reconcile deletions (tombstones) before content.
        let remote_tombs = client.scroll_tombstones()?;
        for (id, _ts) in &remote_tombs {
            let known = self.meta.lock().map(|m| m.is_tombstoned(id)).unwrap_or(false);
            if !known {
                let _ = self.delete(id); // removes local point (if any) + records tombstone
            }
        }
        let local_tombs: HashMap<String, String> = self
            .meta
            .lock()
            .map(|m| m.tombstones().clone())
            .unwrap_or_default();
        if !local_tombs.is_empty() {
            let points: Vec<TombstonePoint> = local_tombs
                .iter()
                .filter_map(|(id, ts)| {
                    Some(TombstonePoint {
                        id_uuid: uuid_string(id).ok()?,
                        memory_id: id.clone(),
                        deleted_at: ts.clone(),
                    })
                })
                .collect();
            let _ = client.upsert_tombstones(&points);
            let uuids: Vec<String> = points.iter().map(|p| p.id_uuid.clone()).collect();
            let _ = client.delete_points(&uuids);
        }
        let tombstoned: std::collections::HashSet<String> = local_tombs.keys().cloned().collect();

        // 2. Content: remote state, keyed by memory id (from payloads).
        let mut remote: HashMap<String, Memory> = HashMap::new();
        for p in client.scroll_all()? {
            if let Ok(m) = serde_json::from_value::<Memory>(p) {
                if !tombstoned.contains(&m.id) {
                    remote.insert(m.id.clone(), m);
                }
            }
        }

        // 3. Local side, through the sync policy.
        let mut local: HashMap<String, Memory> = HashMap::new();
        let mut retract: Vec<String> = Vec::new();
        for m in self.list()? {
            if tombstoned.contains(&m.id) {
                continue;
            }
            let d = self.sync_decision(&m);
            // Re-stamp stale decisions (payload-only), e.g. memories stored
            // before the policy existed.
            let want_state = if d.share { m.sync_state.clone() } else { SyncState::LocalOnly };
            let want_sens = if d.share { Sensitivity::Shareable } else { Sensitivity::LocalOnly };
            if m.sync_reason != d.reason || m.sensitivity != want_sens || m.sync_state != want_state {
                let state = if d.share && want_state == SyncState::LocalOnly {
                    SyncState::Pending
                } else {
                    want_state
                };
                let _ = self.set_payload_fields(
                    &m.id,
                    serde_json::json!({
                        "sync_reason": d.reason, "sensitivity": want_sens, "sync_state": state,
                    }),
                );
            }
            if d.share {
                local.insert(m.id.clone(), m);
            } else {
                if m.kind != MemoryKind::Entity {
                    report.withheld += 1;
                    *report.withheld_by.entry(d.category.clone()).or_insert(0) += 1;
                }
                // Private now: take any cloud copy back, and never pull it.
                if remote.remove(&m.id).is_some() {
                    if let Ok(u) = uuid_string(&m.id) {
                        retract.push(u);
                    }
                    report.retracted += 1;
                }
            }
        }
        if !retract.is_empty() {
            client.delete_points(&retract)?;
        }

        // 4. Reconcile with a 3-way merge against the last-synced version.
        let mut to_push: Vec<Memory> = Vec::new();
        let mut to_pull: Vec<Memory> = Vec::new();
        let mut agreed: Vec<(String, String)> = Vec::new();

        let mut ids: Vec<String> = local.keys().cloned().collect();
        ids.extend(remote.keys().filter(|k| !local.contains_key(*k)).cloned());
        ids.sort();

        for id in ids {
            match (local.get(&id), remote.get(&id)) {
                (Some(l), None) => to_push.push(l.clone()),
                (None, Some(r)) => {
                    if Self::pull_eligible(r, &device, pull_all) {
                        to_pull.push(r.clone());
                    } else {
                        report.cloud_only += 1;
                    }
                }
                (Some(l), Some(r)) if l.updated_at == r.updated_at => {
                    agreed.push((l.id.clone(), l.updated_at.clone()));
                    if l.sync_state != SyncState::Synced {
                        let _ = self.set_payload_fields(
                            &l.id,
                            serde_json::json!({ "sync_state": "synced" }),
                        );
                    }
                }
                (Some(l), Some(r)) => {
                    let base = self.meta.lock().ok().and_then(|m| m.synced_version(&l.id));
                    let local_changed = base.as_deref() != Some(l.updated_at.as_str());
                    let remote_changed = base.as_deref() != Some(r.updated_at.as_str());
                    let same_content = l.text == r.text && l.title == r.title;
                    if !remote_changed || (same_content && l.updated_at > r.updated_at) {
                        to_push.push(l.clone());
                    } else if !local_changed || same_content {
                        to_pull.push(r.clone());
                    } else {
                        // Real conflict: both sides edited since the last sync.
                        // The newer edit wins; the other is kept as an older
                        // version so no information is lost.
                        report.conflicts += 1;
                        let local_wins = l.updated_at > r.updated_at;
                        let (mut winner, loser) =
                            if local_wins { (l.clone(), r) } else { (r.clone(), l) };
                        if is_versionable(&winner.kind) {
                            if let Ok(copy) = self.preserve_conflict_version(loser, &winner.id) {
                                winner.supersedes = Some(copy.id.clone());
                                winner.version = l.version.max(r.version) + 1;
                                to_push.push(copy);
                            }
                        }
                        if local_wins {
                            to_push.push(winner);
                        } else {
                            to_pull.push(winner);
                        }
                    }
                }
                (None, None) => {}
            }
        }

        // 5. Push (batched embedding), marked synced on both sides.
        let mut done: Vec<(String, String)> = agreed;
        if !to_push.is_empty() {
            let pushing: Vec<Memory> = to_push
                .into_iter()
                .map(|mut m| {
                    m.sync_state = SyncState::Synced;
                    m
                })
                .collect();
            client.upsert(&self.to_server_points(&pushing)?)?;
            for m in &pushing {
                // Winners of a conflict may carry a new chain link.
                let fields = serde_json::json!({
                    "sync_state": "synced", "supersedes": m.supersedes, "version": m.version,
                });
                if self.set_payload_fields(&m.id, fields).is_ok() {
                    report.pushed += 1;
                    done.push((m.id.clone(), m.updated_at.clone()));
                }
            }
        }

        // 6. Pull, then rebuild the graph edges pulled memories imply.
        let mut pulled: Vec<Memory> = Vec::new();
        for r in to_pull {
            let mut mm = r;
            mm.sync_state = SyncState::Synced;
            if self.upsert(&mm).is_ok() {
                report.pulled += 1;
                done.push((mm.id.clone(), mm.updated_at.clone()));
                pulled.push(mm);
            }
        }
        if !pulled.is_empty() {
            if let Ok(mut g) = self.graph.lock() {
                for m in &pulled {
                    if let (MemoryKind::DocChunk, Some(p)) = (&m.kind, &m.parent_id) {
                        let _ = g.add_edge(&m.id, p, "part_of");
                    } else if let Some(t) = &m.topic_id {
                        let _ = g.add_in_topic(&m.id, t);
                    }
                    if let Some(prev) = &m.supersedes {
                        let _ = g.add_edge(&m.id, prev, "updates");
                    }
                }
            }
        }

        if let Ok(mut meta) = self.meta.lock() {
            let _ = meta.set_synced_versions(&done);
        }
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(report)
    }

    /// Cloud search (online only): hybrid query against the server for
    /// knowledge that is NOT on this device (other devices' raw memories left
    /// in the cloud by the tiered pull). Short timeout; errors mean "offline".
    pub fn cloud_search(
        &self,
        client: &SyncClient,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let (dense, sparse) = self.embedders.embed_query(query)?;
        let hits = client.query(&dense, &sparse.indices, &sparse.values, limit * 3)?;
        let mut out = Vec::new();
        for (payload, score) in hits {
            let Ok(m) = serde_json::from_value::<Memory>(payload) else { continue };
            // Only what this device does not hold, never outdated versions,
            // and (defensively) never anything the policy would keep local.
            if self.get(&m.id)?.is_some()
                || m.superseded_by.is_some()
                || m.kind == MemoryKind::Topic
                || !self.member_shareable(&m)
            {
                continue;
            }
            out.push(SearchResult { memory: m, score });
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
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
                topic_id: m.topic_id.clone(),
                salience: m.salience,
                superseded: m.superseded_by.is_some(),
                local_only: m.sync_state == SyncState::LocalOnly,
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
        let archived = memories.iter().filter(|m| m.archived).count();
        Ok(GraphData {
            nodes,
            edges,
            archived,
        })
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
                    // Summaries sync: never feed them private memories.
                    if m.kind == MemoryKind::Summary || !self.member_shareable(m) {
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
                    topic_id: None,
                    due_at: None,
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
        // Re-home stragglers, then refresh subjects that gathered new members.
        let topics_rehomed = self.consolidate_topics().unwrap_or(0);
        let topics_summarized = self.refresh_due_topics(model, 3, 12);
        let _ = self.prune_empty_topics();
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
            topics_summarized,
            topics_rehomed,
        })
    }

    pub fn clear_archive(&self) -> Result<usize> {
        self.meta
            .lock()
            .map_err(|_| anyhow!("meta poisoned"))?
            .clear_archive()
    }

    // ---- scheduling (timeline / calendar) -------------------------------

    /// Active memories that have a due date, earliest first.
    pub fn list_scheduled(&self) -> Result<Vec<Memory>> {
        // Outdated versions (a rescheduled deadline's old date) are hidden.
        let mut items: Vec<Memory> = self
            .list_active()?
            .into_iter()
            .filter(|m| m.due_at.as_deref().map(|d| !d.is_empty()).unwrap_or(false))
            .filter(|m| m.superseded_by.is_none())
            .collect();
        items.sort_by(|a, b| a.due_at.cmp(&b.due_at));
        Ok(items)
    }

    /// Mark a scheduled item done / not-done (payload-only, no re-embed).
    pub fn set_done(&self, id: &str, done: bool) -> Result<()> {
        self.set_payload_fields(
            id,
            serde_json::json!({ "done": done, "updated_at": chrono::Utc::now().to_rfc3339() }),
        )?;
        self.shard.flush().map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(())
    }

    // ---- retrieval -------------------------------------------------------

    pub fn search(&self, req: SearchRequest) -> Result<SearchResponse> {
        let started = Instant::now();
        let (dense, sparse) = self.embedders.embed_query(&req.query)?;
        let filter = build_filter(req.site_id.as_deref(), req.kind.as_ref());

        // Fetch a wider candidate set so we can diversify + rerank down to limit.
        let candidate_limit = req.limit.saturating_mul(5).clamp(req.limit, 90);
        let mut builder = QueryRequestBuilder::new(candidate_limit)
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

        let mut results = self.hydrate(
            scored
                .into_iter()
                .filter_map(|sp| Some((memory_from_payload(sp.payload.as_ref())?, sp.score)))
                .collect(),
        );

        // Rerank: relevance dominates; salience only gently breaks ties.
        if results.len() > 1 {
            let max = results.iter().map(|r| r.score).fold(f32::MIN, f32::max);
            let min = results.iter().map(|r| r.score).fold(f32::MAX, f32::min);
            let range = (max - min).max(1e-6);
            results.sort_by(|a, b| {
                let ba = 0.9 * ((a.score - min) / range) + 0.1 * a.memory.salience;
                let bb = 0.9 * ((b.score - min) / range) + 0.1 * b.memory.salience;
                bb.partial_cmp(&ba).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Diversify: cap results from any one source document so a single doc
        // can't monopolize the context (e.g. a study guide drowning your resume).
        if results.len() > req.limit {
            let cap = 3usize;
            let mut per_doc: HashMap<String, usize> = HashMap::new();
            let mut primary: Vec<SearchResult> = Vec::new();
            let mut overflow: Vec<SearchResult> = Vec::new();
            for r in results {
                let key = r.memory.parent_id.clone().unwrap_or_else(|| r.memory.id.clone());
                let c = per_doc.entry(key).or_insert(0);
                if *c < cap {
                    *c += 1;
                    primary.push(r);
                } else {
                    overflow.push(r);
                }
            }
            primary.extend(overflow);
            primary.truncate(req.limit);
            results = primary;
        }

        Ok(SearchResponse {
            results,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            mode: req.mode,
        })
    }

    /// Chat retrieval: ranked evidence for `query` (see `retrieve_context`).
    pub fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        Ok(self.retrieve_context(query, limit)?.results)
    }

    /// Topic-aware retrieval (schema first, then details — how recall works):
    ///   1. Base: hybrid dense + BM25 search over content (hubs excluded).
    ///   2. Activate topics: the subjects the evidence belongs to (rank-weighted
    ///      votes from the base hits) plus direct matches on topic nodes.
    ///   3. Pull the members of each active topic (a note that shares no words
    ///      with the query still surfaces when its subject is active).
    ///   4. Documents the query names ("my resume", "coresum", "HCMA"), and the
    ///      user's profile for questions about the user as a person.
    ///   5. Weighted reciprocal-rank fusion of all lists; when one subject
    ///      clearly owns the query, results outside it are demoted.
    ///   6. Per-document diversity cap. Scores are normalized to [0, 1].
    pub fn retrieve_context(&self, query: &str, limit: usize) -> Result<RetrievalContext> {
        let t = self.tuning;
        let (dense, sparse) = self.embedders.embed_query(query)?;
        let pool = (limit * 5).max(30);

        let base = self.query_points(&dense, Some(&sparse), Some(content_filter()), pool)?;
        let active = self.activate_topics(&dense, &base);
        let active_ids: std::collections::HashSet<&str> =
            active.iter().map(|a| a.0.as_str()).collect();

        // The strongest direct evidence is never demoted by a topic vote: a
        // precise hit ("5k") must not lose to many weak same-topic matches.
        let protected: Vec<String> = base.iter().take(2).map(|r| r.memory.id.clone()).collect();
        let top_hit: Option<SearchResult> = base.first().cloned();

        // (ranked list, fusion weight, anchored = exempt from off-topic demotion)
        let mut lists: Vec<(Vec<SearchResult>, f32, bool)> = vec![(base, 1.0, false)];
        for (tid, share) in &active {
            let mut f = content_filter();
            f.must = Some(vec![match_cond("topic_id", tid)]);
            if let Ok(members) = self.query_points(&dense, Some(&sparse), Some(f), limit * 2) {
                lists.push((members, t.topic_weight * share, false));
            }
        }
        let named = self.referenced_documents(query).unwrap_or_default();
        for d in &named {
            if let Ok(h) = self.search_in_doc(&dense, &sparse, d, 4) {
                lists.push((h, t.named_weight, true));
            }
        }
        if needs_profile(query) {
            for d in self.profile_documents().unwrap_or_default() {
                if named.contains(&d) {
                    continue;
                }
                if let Ok(h) = self.search_in_doc(&dense, &sparse, &d, 3) {
                    lists.push((h, t.profile_weight, true));
                }
            }
        }

        // Weighted reciprocal-rank fusion.
        let mut fused: HashMap<String, f32> = HashMap::new();
        let mut anchored: std::collections::HashSet<String> = protected.into_iter().collect();
        let mut best: HashMap<String, SearchResult> = HashMap::new();
        for (list, w, anchor) in lists {
            for (rank, r) in list.into_iter().enumerate() {
                *fused.entry(r.memory.id.clone()).or_insert(0.0) +=
                    w / (t.rrf_k + rank as f32 + 1.0);
                if anchor {
                    anchored.insert(r.memory.id.clone());
                }
                best.entry(r.memory.id.clone()).or_insert(r);
            }
        }

        let focused = active.first().map(|a| a.1 >= t.focus_conf).unwrap_or(false);
        let mut ranked: Vec<SearchResult> = best
            .into_values()
            .map(|mut r| {
                let mut s = fused.get(&r.memory.id).copied().unwrap_or(0.0);
                if focused && !anchored.contains(&r.memory.id) {
                    let on_topic = r
                        .memory
                        .topic_id
                        .as_deref()
                        .map(|x| active_ids.contains(x))
                        .unwrap_or(false);
                    if !on_topic {
                        s *= t.offtopic_penalty;
                    }
                }
                // Salience only breaks near-ties.
                r.score = s * (1.0 + 0.05 * r.memory.salience);
                r
            })
            .collect();
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Evolving memory: unless the question is about the past, an outdated
        // version is replaced by its current version ("exam is Friday" ->
        // "exam moved to Monday"), keeping the rank it earned.
        let history = is_history_query(query);
        let mut top_hit = top_hit;
        if !history {
            let mut seen = std::collections::HashSet::new();
            let mut current = Vec::with_capacity(ranked.len());
            for r in ranked {
                let r = if r.memory.superseded_by.is_some() {
                    SearchResult {
                        memory: self.latest_version(r.memory),
                        score: r.score,
                    }
                } else {
                    r
                };
                if seen.insert(r.memory.id.clone()) {
                    current.push(r);
                }
            }
            ranked = current;
            top_hit = top_hit.map(|r| SearchResult {
                memory: self.latest_version(r.memory),
                score: r.score,
            });
        }
        let mut results = diversify(ranked, limit, t.per_doc_cap);
        // Guarantee the single best raw hit a slot: fusion may reorder evidence
        // but must never drop the strongest direct match entirely.
        if let Some(top) = top_hit {
            if !results.iter().any(|r| r.memory.id == top.memory.id) {
                if results.len() >= limit {
                    results.pop();
                }
                let s = results.last().map(|r| r.score).unwrap_or(0.0);
                results.push(SearchResult { score: s, ..top });
            }
        }
        if let Some(top) = results.first().map(|r| r.score).filter(|s| *s > 0.0) {
            for r in &mut results {
                r.score /= top;
            }
        }

        let topics = active
            .iter()
            .filter_map(|(tid, share)| {
                let m = self.get(tid).ok().flatten()?;
                Some(TopicHit {
                    id: tid.clone(),
                    summary: topic_summary(&m),
                    name: m.title,
                    confidence: *share,
                })
            })
            .collect();
        Ok(RetrievalContext { results, topics })
    }

    /// Which subjects does this query belong to? Base hits vote for their topic
    /// (rank-weighted), and topic nodes that match the query directly add to the
    /// vote. Returns up to two topics with their share of the vote, strongest
    /// first, filtered by `activation_min`.
    fn activate_topics(&self, dense: &[f32], base: &[SearchResult]) -> Vec<(String, f32)> {
        let t = self.tuning;
        // Votes decay with the square of rank: the subject of the BEST evidence
        // wins, not the subject with the most (weak) matches.
        let mut vote: HashMap<String, f32> = HashMap::new();
        for (i, r) in base.iter().take(12).enumerate() {
            if let Some(tid) = &r.memory.topic_id {
                *vote.entry(tid.clone()).or_insert(0.0) += 1.0 / ((i + 1) * (i + 1)) as f32;
            }
        }
        if let Ok(hits) = self.query_points(dense, None, Some(kind_filter(MemoryKind::Topic)), 3) {
            for h in hits {
                if h.score >= t.topic_join_direct {
                    *vote.entry(h.memory.id).or_insert(0.0) += 0.2 + (h.score - t.topic_join_direct);
                }
            }
        }
        let total: f32 = vote.values().sum();
        if total <= 0.0 {
            return Vec::new();
        }
        let mut v: Vec<(String, f32)> = vote.into_iter().map(|(k, x)| (k, x / total)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.into_iter().filter(|(_, s)| *s >= t.activation_min).take(2).collect()
    }

    /// One memory by id (None if missing or deleted).
    pub fn get(&self, id: &str) -> Result<Option<Memory>> {
        let req = RetrieveRequestBuilder::new(vec![point_id(id)?])
            .with_payload(WithPayloadInterface::Bool(true))
            .build();
        let recs = self
            .shard
            .retrieve(req)
            .map_err(|e| anyhow!("retrieve failed: {e}"))?;
        let tomb = self.meta.lock().map(|m| m.is_tombstoned(id)).unwrap_or(false);
        if tomb {
            return Ok(None);
        }
        Ok(recs.into_iter().find_map(|r| memory_from_payload(r.payload.as_ref())))
    }

    /// Documents that describe the user themselves (resume / CV / about-me).
    fn profile_documents(&self) -> Result<Vec<String>> {
        Ok(self
            .list_documents()?
            .into_iter()
            .filter(|m| is_profile_title(&m.title))
            .map(|m| m.id)
            .collect())
    }

    /// Documents the query refers to by name. Matching is on whole tokens (no
    /// substrings: "for" must not match "In-for-med"), with compound joins
    /// ("core-sum" -> "coresum", "study guide" -> "studyguide") and title
    /// acronyms ("HCMA"). Short, name-like titles need one matching token; long
    /// descriptive titles need two. More than two hits = no specific reference.
    fn referenced_documents(&self, query: &str) -> Result<Vec<String>> {
        let qtok = query_tokens(query);
        if qtok.is_empty() {
            return Ok(Vec::new());
        }
        let mut refs = Vec::new();
        for m in self.list_documents()? {
            let tt = title_tokens(&m.title);
            if tt.is_empty() {
                continue;
            }
            let hits = tt.iter().filter(|w| qtok.contains(*w)).count();
            let acronym = title_acronym_hit(&tt, &qtok);
            let needed = if tt.len() <= 3 { 1 } else { 2 };
            if acronym || hits >= needed {
                refs.push(m.id);
            }
        }
        if refs.len() > 2 {
            return Ok(Vec::new());
        }
        Ok(refs)
    }

    /// Hybrid search restricted to the chunks of one document.
    fn search_in_doc(
        &self,
        dense: &[f32],
        sparse: &qdrant_edge::SparseVector,
        doc_id: &str,
        top: usize,
    ) -> Result<Vec<SearchResult>> {
        let f = Filter {
            must: Some(vec![match_cond("parent_id", doc_id)]),
            ..Default::default()
        };
        self.query_points(dense, Some(sparse), Some(f), top)
    }

    /// Core vector query. Hybrid (dense + BM25 sparse, fused with RRF) when
    /// `sparse` is given, dense-only otherwise; `filter` applies to every stage.
    /// Results are hydrated (archived/tombstoned dropped, live salience set) and
    /// carry raw scores (cosine for dense-only).
    fn query_points(
        &self,
        dense: &[f32],
        sparse: Option<&qdrant_edge::SparseVector>,
        filter: Option<Filter>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut builder =
            QueryRequestBuilder::new(limit).with_payload(WithPayloadInterface::Bool(true));
        builder = match sparse {
            Some(sp) => {
                let mut pd = PrefetchBuilder::new(PREFETCH_LIMIT)
                    .query(nearest(DENSE_NAME, VectorInternal::Dense(dense.to_vec())));
                let mut ps = PrefetchBuilder::new(PREFETCH_LIMIT)
                    .query(nearest(SPARSE_NAME, VectorInternal::Sparse(sp.clone())));
                if let Some(f) = &filter {
                    pd = pd.filter(f.clone());
                    ps = ps.filter(f.clone());
                }
                builder
                    .add_prefetch(pd.build())
                    .add_prefetch(ps.build())
                    .query(ScoringQuery::Fusion(Fusion::Rrf {
                        k: 60,
                        weights: None,
                    }))
            }
            None => {
                let b = builder.query(nearest(DENSE_NAME, VectorInternal::Dense(dense.to_vec())));
                match &filter {
                    Some(f) => b.filter(f.clone()),
                    None => b,
                }
            }
        };
        let scored = self
            .shard
            .query(builder.build())
            .map_err(|e| anyhow!("query failed: {e}"))?;
        Ok(self.hydrate(
            scored
                .into_iter()
                .filter_map(|sp| Some((memory_from_payload(sp.payload.as_ref())?, sp.score)))
                .collect(),
        ))
    }

    /// Drop archived/tombstoned hits and attach live salience. Takes each
    /// side-store lock once and reads per-hit values (no per-query cloning of
    /// the archive / tombstone / access sets).
    fn hydrate(&self, hits: Vec<(Memory, f32)>) -> Vec<SearchResult> {
        let hits: Vec<(Memory, f32, u32)> = match self.meta.lock() {
            Ok(m) => hits
                .into_iter()
                .filter(|(mem, _)| !m.is_archived(&mem.id) && !m.is_tombstoned(&mem.id))
                .map(|(mem, s)| {
                    let a = m.access_count(&mem.id);
                    (mem, s, a)
                })
                .collect(),
            Err(_) => hits.into_iter().map(|(m, s)| (m, s, 0)).collect(),
        };
        let graph = self.graph.lock().ok();
        hits.into_iter()
            .map(|(mut memory, score, access)| {
                let deg = graph.as_ref().map(|g| g.degree(&memory.id)).unwrap_or(0);
                memory.salience = salience_of(&memory, deg, access);
                SearchResult { memory, score }
            })
            .collect()
    }

    /// All live memories of one kind, via the indexed `kind` payload filter
    /// (avoids scrolling and deserializing the whole store).
    pub fn list_kind(&self, kind: MemoryKind) -> Result<Vec<Memory>> {
        self.scroll_where(kind_filter(kind))
    }

    /// Live memories matching a payload filter (indexed), newest first.
    fn scroll_where(&self, filter: Filter) -> Result<Vec<Memory>> {
        let req = ScrollRequestBuilder::new()
            .limit(SCROLL_ALL_LIMIT)
            .filter(filter)
            .with_payload(WithPayloadInterface::Bool(true))
            .build();
        let (records, _next) = self
            .shard
            .scroll(req)
            .map_err(|e| anyhow!("scroll failed: {e}"))?;
        let mut out: Vec<Memory> = {
            let meta = self.meta.lock().map_err(|_| anyhow!("meta poisoned"))?;
            records
                .into_iter()
                .filter_map(|r| memory_from_payload(r.payload.as_ref()))
                .filter(|m| !meta.is_tombstoned(&m.id))
                .map(|mut m| {
                    m.archived = meta.is_archived(&m.id);
                    m
                })
                .collect()
        };
        out.sort_by(|a, b| b.id.cmp(&a.id));
        Ok(out)
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
        let (archived, tombstoned, access) = self
            .meta
            .lock()
            .map(|m| {
                (
                    m.archived_ids().clone(),
                    m.tombstones()
                        .keys()
                        .cloned()
                        .collect::<std::collections::HashSet<_>>(),
                    m.access_snapshot(),
                )
            })
            .unwrap_or_default();
        // A tombstoned id should never surface (defensive).
        memories.retain(|m| !tombstoned.contains(&m.id));
        let deg = self.degrees();
        for m in &mut memories {
            m.archived = archived.contains(&m.id);
            m.salience = salience_of(
                m,
                deg.get(&m.id).copied().unwrap_or(0),
                access.get(&m.id).copied().unwrap_or(0),
            );
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
        self.graph.lock().map(|g| g.degrees()).unwrap_or_default()
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

/// Minimum similarity for an older memory to be considered as the one a new
/// memory updates. Kept low so the LLM judge can see world-knowledge revisions
/// ("live in Pune" -> "moved to Bangalore"); the rules have their own, much
/// higher thresholds.
const VERSION_MIN_SIM: f32 = 0.30;

fn is_versionable(kind: &MemoryKind) -> bool {
    matches!(
        kind,
        MemoryKind::Note | MemoryKind::Observation | MemoryKind::Event | MemoryKind::Measurement
    )
}

/// Words signalling that a statement revises an earlier one.
const UPDATE_CUES: &[&str] = &[
    "moved to", "moved", "changed", "change to", "now", "no longer", "instead", "updated",
    "actually", "rescheduled", "switched", "postponed", "pushed to", "pushed back",
    "not anymore", "anymore", "cancelled", "canceled", "new", "improved", "dropped", "raised",
    "increased", "decreased", "went up", "went down", "is now", "correction",
];

/// Deterministic update detection (used without an LLM, and as a fallback).
/// Conservative: requires high similarity AND a revision signal.
///   - events: same title, different due date;
///   - otherwise: similarity >= 0.62 with an update cue ("moved to", "now",
///     "switched") and a shared content word, or similarity >= 0.72 with a
///     changed number/date and >= 2 shared content words.
fn heuristic_updates(old: &Memory, new: &Memory, sim: f32) -> bool {
    if old.kind == MemoryKind::Event && new.kind == MemoryKind::Event {
        let same_title = content_words(&old.title) == content_words(&new.title)
            && !content_words(&new.title).is_empty();
        if same_title && old.due_at.is_some() && old.due_at != new.due_at {
            return true;
        }
    }
    let lower = new.text.to_lowercase();
    let padded = format!(" {} ", lower.replace(|c: char| !c.is_alphanumeric(), " "));
    let cue = UPDATE_CUES
        .iter()
        .any(|c| padded.contains(&format!(" {c} ")));
    let a = content_words(&old.text);
    let b = content_words(&new.text);
    let shared = a.intersection(&b).count();
    let nums_old = numbers_in(&old.text);
    let nums_new = numbers_in(&new.text);
    let changed_number = !nums_new.is_empty() && !nums_old.is_empty() && nums_new != nums_old;
    // An explicit revision cue on a closely similar statement needs only one
    // shared subject word ("My EXAM is Friday" -> "My EXAM got moved to
    // Monday"); a bare number change needs more agreement.
    (sim >= 0.62 && cue && shared >= 1) || (sim >= 0.72 && changed_number && shared >= 2)
}

/// Lowercase content words (>= 3 chars, not stopwords).
fn content_words(text: &str) -> std::collections::HashSet<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has",
        "have", "had", "but", "not", "you", "your", "our", "its", "into", "got", "been", "will",
        "now", "moved", "changed", "new", "instead", "actually", "task", "event",
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3 && !STOP.contains(w))
        .map(String::from)
        .collect()
}

/// Numbers and dates mentioned in text (for "a value changed" detection).
fn numbers_in(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '/'))
        .map(|t| t.trim_matches(|c| c == '.' || c == '-' || c == '/'))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .map(String::from)
        .collect()
}

/// Does the question ask about the past ("what was it before", "used to")?
/// Then outdated versions are kept in the answer's context.
fn is_history_query(query: &str) -> bool {
    let q = format!(" {} ", query.to_lowercase());
    [
        " before", " previously", " used to ", " originally", " history", " changed",
        " earlier", " old ", " last time", " first ", " was it ", " were they ", " past ",
    ]
    .iter()
    .any(|c| q.contains(c))
}

fn match_cond(field: &str, value: &str) -> Condition {
    Condition::Field(FieldCondition::new_match(jpath(field), Match::from(value.to_string())))
}

/// Content only: excludes organizing hubs (entity names, topic nodes).
fn content_filter() -> Filter {
    Filter {
        must_not: Some(vec![match_cond("kind", "entity"), match_cond("kind", "topic")]),
        ..Default::default()
    }
}

fn kind_filter(kind: MemoryKind) -> Filter {
    Filter {
        must: Some(vec![match_cond("kind", &kind_str(&kind))]),
        ..Default::default()
    }
}

/// Keep at most `cap` results per source document, then truncate to `limit`.
/// Over-cap results only fill slots left over, so one document can't
/// monopolize the context.
fn diversify(results: Vec<SearchResult>, limit: usize, cap: usize) -> Vec<SearchResult> {
    let mut per_doc: HashMap<String, usize> = HashMap::new();
    let mut primary: Vec<SearchResult> = Vec::new();
    let mut overflow: Vec<SearchResult> = Vec::new();
    for r in results {
        let key = r.memory.parent_id.clone().unwrap_or_else(|| r.memory.id.clone());
        let c = per_doc.entry(key).or_insert(0);
        if *c < cap {
            *c += 1;
            primary.push(r);
        } else {
            overflow.push(r);
        }
    }
    primary.extend(overflow);
    primary.truncate(limit);
    primary
}

/// Position of a chunk within its document, from its "name [n]" title.
fn chunk_index(title: &str) -> usize {
    title
        .rsplit('[')
        .next()
        .and_then(|s| s.trim_end_matches(']').trim().parse().ok())
        .unwrap_or(usize::MAX)
}

/// A topic node's rolling summary ("" until one has been written; the node's
/// text equals its name before that).
fn topic_summary(m: &Memory) -> String {
    if m.text.trim() == m.title.trim() {
        String::new()
    } else {
        m.text.clone()
    }
}

/// Is this document about the user themselves (resume / CV / about-me)?
fn is_profile_title(title: &str) -> bool {
    let t = title.to_lowercase();
    const HINTS: &[&str] = &[
        "resume", "resumé", "résumé", "curriculum vitae", "curriculum-vitae", "about me",
        "about-me", "aboutme", "biodata",
    ];
    // "bio" / "profile" alone are ambiguous ("Bio 101 notes", "Company Profile"),
    // so only unambiguous identity markers count.
    HINTS.iter().any(|h| t.contains(h)) || t.split(|c: char| !c.is_alphanumeric()).any(|w| w == "cv")
}

/// Should the user's profile ground this answer? Only for questions about the
/// user AS A PERSON:
///   - person cues: "my strengths", "my skills", "my projects", "my background";
///   - fit questions: an evaluative word + a career word ("best internships for
///     me", "what jobs suit me", "should I apply to this role").
/// Not for procedures ("how do I keep my F-1 status while working an
/// internship", "when can I apply for OPT") and not for every "my" ("what is my
/// coresum score" is about CoreSum).
fn needs_profile(query: &str) -> bool {
    if !is_self_referential(query) {
        return false;
    }
    const PERSON: &[&str] = &[
        "skill", "skills", "strength", "strengths", "weakness", "weaknesses", "experience",
        "background", "education", "qualified", "qualification", "qualifications", "resume",
        "cv", "profile", "portfolio", "expertise", "achievements", "projects",
        "accomplishments",
    ];
    const CAREER: &[&str] = &[
        "intern", "internship", "internships", "job", "jobs", "career", "careers", "role",
        "roles", "position", "positions", "company", "companies", "program", "programs",
        "major", "field", "fields", "path",
    ];
    const EVAL: &[&str] = &[
        "best", "suit", "suits", "suited", "fit", "fits", "good", "right", "recommend",
        "match", "matches", "ideal", "should",
    ];
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
    let has = |set: &[&str]| words.iter().any(|w| set.contains(w));
    has(PERSON) || (has(CAREER) && has(EVAL))
}

/// Filler words that never identify a document by title.
const GENERIC_TITLE_WORDS: &[&str] = &[
    "pdf", "doc", "docx", "txt", "md", "the", "and", "for", "with", "from", "into", "file",
    "files", "final", "copy", "new", "old", "version", "draft", "notes", "note", "document",
    "documents", "report", "paper", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
];

/// Identifying tokens of a document title, in order.
fn title_tokens(title: &str) -> Vec<String> {
    title
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            w.chars().count() >= 3
                && !w.chars().all(|c| c.is_ascii_digit())
                && !GENERIC_TITLE_WORDS.contains(w)
        })
        .map(String::from)
        .collect()
}

/// Query tokens plus adjacent-pair joins, so "core-sum" also yields "coresum"
/// and "study guide" also yields "studyguide".
fn query_tokens(query: &str) -> std::collections::HashSet<String> {
    let words: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    let mut out: std::collections::HashSet<String> = words.iter().cloned().collect();
    for pair in words.windows(2) {
        out.insert(format!("{}{}", pair[0], pair[1]));
    }
    out
}

/// Does the query use the title's acronym ("HCMA" for "Hierarchical Cognitive
/// Memory Architecture ...")? Prefixes of 4+ letters count; a 3-letter acronym
/// must match exactly.
fn title_acronym_hit(tt: &[String], qtok: &std::collections::HashSet<String>) -> bool {
    let acr: String = tt.iter().filter_map(|w| w.chars().next()).collect();
    let n = acr.chars().count();
    if n < 3 {
        return false;
    }
    if n == 3 {
        return qtok.contains(&acr);
    }
    (4..=n).any(|len| qtok.contains(&acr.chars().take(len).collect::<String>()))
}

/// Normalize a topic label (from a model or a file name): first non-empty line,
/// quotes/punctuation stripped, at most 4 words, no dangling connector words
/// ("... Architecture for"), and shouting converted to Title Case (short
/// acronyms like TCS / OPT are kept).
fn clean_label(s: &str) -> String {
    const CONNECTORS: &[&str] = &["for", "and", "of", "the", "in", "on", "with", "to", "a", "an", "&"];
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    let cleaned = line.trim_matches(|c: char| {
        c == '"' || c == '\'' || c == '.' || c == ':' || c == '-' || c == '*' || c == '#'
    });
    let mut words: Vec<&str> = cleaned.split_whitespace().take(4).collect();
    while words
        .last()
        .map(|w| CONNECTORS.contains(&w.to_lowercase().as_str()))
        .unwrap_or(false)
    {
        words.pop();
    }
    let letters: String = words.concat().chars().filter(|c| c.is_alphabetic()).collect();
    let shouting = letters.chars().count() > 4 && letters.chars().all(|c| c.is_uppercase());
    words
        .iter()
        .map(|w| {
            if shouting && w.chars().count() > 3 {
                title_word(w)
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_word(w: &str) -> String {
    let mut cs = w.chars();
    match cs.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &cs.as_str().to_lowercase(),
        None => String::new(),
    }
}

/// A topic name from a document's file name: "Usa_StudyGuide.pdf" ->
/// "Usa Study Guide", "Weeknight Recipes.txt" -> "Weeknight Recipes". Drops the
/// extension, dates, and version noise. No LLM needed; the rolling summary may
/// refine it later.
fn name_from_title(title: &str) -> String {
    let stem = match title.rsplit_once('.') {
        Some((s, ext)) if ext.len() <= 5 && !ext.contains(' ') => s,
        _ => title,
    };
    // Split camelCase ("StudyGuide" -> "Study Guide") and separators.
    let mut spaced = String::new();
    let mut prev_lower = false;
    for c in stem.chars() {
        if c.is_uppercase() && prev_lower {
            spaced.push(' ');
        }
        prev_lower = c.is_lowercase();
        spaced.push(if c.is_alphanumeric() { c } else { ' ' });
    }
    let words: Vec<String> = spaced
        .split_whitespace()
        .filter(|w| {
            let l = w.to_lowercase();
            !w.chars().all(|c| c.is_ascii_digit())
                && !GENERIC_TITLE_WORDS.contains(&l.as_str())
                && l != "v"
        })
        .map(|w| {
            if w.chars().all(|c| c.is_lowercase()) {
                title_word(w)
            } else {
                w.to_string()
            }
        })
        .collect();
    clean_label(&words.join(" "))
}

/// Fallback topic label: the first few significant words of the content, in
/// Title Case.
fn heuristic_label(text: &str) -> String {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "this", "that", "from", "your", "you", "are", "was",
        "were", "have", "has", "about", "into", "over", "will", "would", "should",
    ];
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4 && !STOP.contains(&w.to_lowercase().as_str()))
        .take(3)
        .map(|w| {
            let mut cs = w.chars();
            match cs.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &cs.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect();
    if words.is_empty() {
        "General".to_string()
    } else {
        words.join(" ")
    }
}

/// Does the query ask about the user themselves? Detects first-person pronouns
/// as whole words ("what internships are best for me", "my strengths", "should
/// I ...") so we ground the answer in the user's own profile documents.
fn is_self_referential(query: &str) -> bool {
    const PRONOUNS: &[&str] = &["me", "my", "mine", "myself", "i", "im", "id", "ive"];
    let lower = query.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| PRONOUNS.contains(&w))
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

/// Salience in [0,1]: recency + graph connectivity + usage + kind weight.
fn salience_of(m: &Memory, degree: usize, access: u32) -> f32 {
    let recency = (1.0 / (1.0 + age_days(&m.updated_at) / 30.0)) as f32;
    let deg = (degree as f32 / 4.0).min(1.0);
    let used = (access as f32 / 5.0).min(1.0);
    let kindw = match m.kind {
        MemoryKind::Note
        | MemoryKind::Observation
        | MemoryKind::Event
        | MemoryKind::Measurement
        | MemoryKind::Summary => 1.0,
        MemoryKind::Topic => 0.95,
        MemoryKind::Entity => 0.9,
        MemoryKind::Document => 0.85,
        MemoryKind::DocChunk => 0.5,
    };
    0.42 * recency + 0.25 * deg + 0.18 * used + 0.15 * kindw
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

#[cfg(test)]
mod tests {
    use super::{clean_label, heuristic_label, is_self_referential};

    #[test]
    fn self_reference_detection() {
        assert!(is_self_referential("what type of internships are best for me?"));
        assert!(is_self_referential("what are my strengths"));
        assert!(is_self_referential("should I apply"));
        assert!(!is_self_referential("what is the TCS gap in coresum"));
        assert!(!is_self_referential("summarize the study guide"));
        // "i" must be a whole word, not a letter inside another word.
        assert!(!is_self_referential("kepler mission timeline"));
    }

    #[test]
    fn labels_are_cleaned() {
        assert_eq!(clean_label("\"Exoplanet Transit Research\"\n"), "Exoplanet Transit Research");
        assert_eq!(clean_label("\n\n**Piano Practice**."), "Piano Practice");
        assert_eq!(clean_label("one two three four five six seven").split(' ').count(), 4);
        assert_eq!(clean_label("VIDEO QUALITY ISSUES"), "Video Quality Issues");
        assert_eq!(clean_label("TCS GAP ANALYSIS"), "TCS GAP Analysis");
        assert_eq!(
            clean_label("Hierarchical Cognitive Memory Architecture for Hybrid"),
            "Hierarchical Cognitive Memory Architecture"
        );
        assert_eq!(clean_label("Memory Architecture for"), "Memory Architecture");
    }

    #[test]
    fn profile_titles_are_unambiguous() {
        use super::is_profile_title;
        assert!(is_profile_title("Resume (Aug 2026) (1).pdf"));
        assert!(is_profile_title("Yash_CV_2026.pdf"));
        assert!(is_profile_title("About Me.md"));
        assert!(!is_profile_title("Bio 101 lecture notes.pdf"));
        assert!(!is_profile_title("Company Profile.pdf"));
        assert!(!is_profile_title("Service Level Agreement.pdf"));
    }

    #[test]
    fn document_titles_become_topic_names() {
        use super::name_from_title;
        assert_eq!(name_from_title("Usa_StudyGuide.pdf"), "Usa Study Guide");
        assert_eq!(name_from_title("Weeknight Recipes.txt"), "Weeknight Recipes");
        assert_eq!(name_from_title("coresum.pdf"), "Coresum");
        assert_eq!(name_from_title("Resume (Aug 2026) (1).pdf"), "Resume");
    }

    #[test]
    fn profile_only_for_questions_about_the_person() {
        use super::needs_profile;
        assert!(needs_profile("what type of internships are best for me?"));
        assert!(needs_profile("what are my strengths as an engineer?"));
        assert!(needs_profile("tell me about my projects"));
        assert!(needs_profile("which jobs suit me"));
        // Procedures and topical "my" are not about the person.
        assert!(!needs_profile("how do I keep my F-1 status while working an internship?"));
        assert!(!needs_profile("what is OPT and when can I apply?"));
        assert!(!needs_profile("what is my coresum TCS score right now?"));
        assert!(!needs_profile("how fast is my 5k?"));
    }

    #[test]
    fn document_references_use_whole_tokens() {
        use super::{query_tokens, title_acronym_hit, title_tokens};
        // "for" must not match "In-for-med".
        let tt = title_tokens("Physics-Informed Transit Detection.pdf");
        assert!(!tt.iter().any(|w| query_tokens("internships best for me").contains(w)));
        // Compound joins.
        assert!(query_tokens("benchmarks for core-sum").contains("coresum"));
        assert!(query_tokens("the study guide").contains("studyguide"));
        assert_eq!(title_tokens("Usa_StudyGuide.pdf"), vec!["usa", "studyguide"]);
        assert_eq!(title_tokens("Resume (Aug 2026) (1).pdf"), vec!["resume"]);
        // Acronyms.
        let h = title_tokens("Hierarchical Cognitive Memory Architecture for Hybrid Retrieval.pdf");
        assert!(title_acronym_hit(&h, &query_tokens("how does consolidation work in HCMA?")));
        assert!(!title_acronym_hit(&h, &query_tokens("how does memory work?")));
    }

    #[test]
    fn heuristic_label_falls_back() {
        assert_eq!(heuristic_label("the and for"), "General");
        let l = heuristic_label("Practicing piano scales every day");
        assert!(l.starts_with("Practicing"), "{l}");
    }
}
