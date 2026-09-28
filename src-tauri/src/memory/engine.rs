//! The on-device memory engine: an embedded Qdrant Edge shard plus local
//! embedders. All operations are synchronous, in-process, and offline.

use std::path::Path;
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
use super::record::{
    Memory, MemoryKind, NewMemory, SearchMode, SearchRequest, SearchResponse, SearchResult, Stats,
    SyncState,
};

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

        let engine = Self { shard, embedders };
        engine.ensure_indexes();
        Ok(engine)
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
            sync_state: SyncState::LocalOnly,
            version: 1,
        };
        self.upsert(&memory)?;
        self.shard
            .flush()
            .map_err(|e| anyhow!("flush failed: {e}"))?;
        Ok(memory)
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
        Ok(())
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
