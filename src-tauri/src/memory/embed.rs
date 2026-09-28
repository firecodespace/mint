//! On-device embeddings, fully local:
//!  - Dense: fastembed (ONNX, all-MiniLM-L6-v2, 384d). Model downloaded once on
//!    first run into the cache dir, then offline forever.
//!  - Sparse: Qdrant Edge's built-in BM25 (no extra model, no network ever).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use qdrant_edge::bm25_embed::{EdgeBm25, EdgeBm25Config};
use qdrant_edge::SparseVector;

/// Dimension of all-MiniLM-L6-v2 dense vectors.
pub const DENSE_DIM: usize = 384;

pub struct Embedders {
    // `TextEmbedding::embed` takes `&mut self`, so guard it.
    dense: Mutex<TextEmbedding>,
    sparse: EdgeBm25,
}

impl Embedders {
    /// Initialize embedders. `cache_dir` holds the downloaded ONNX model so the
    /// app is offline after the first run.
    pub fn new(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir).ok();

        let dense = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_cache_dir(cache_dir.to_path_buf())
                .with_show_download_progress(true),
        )
        .context("failed to initialize fastembed dense model")?;

        let sparse =
            EdgeBm25::new(EdgeBm25Config::default()).context("failed to initialize BM25 model")?;

        Ok(Self {
            dense: Mutex::new(dense),
            sparse,
        })
    }

    fn embed_dense(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self
            .dense
            .lock()
            .map_err(|_| anyhow::anyhow!("dense embedder mutex poisoned"))?;
        let mut out = model
            .embed(vec![text], None)
            .context("dense embedding failed")?;
        out.pop().context("dense embedding returned no vectors")
    }

    /// Embed a document (to store). Sparse uses BM25 document (TF) weighting.
    pub fn embed_document(&self, text: &str) -> Result<(Vec<f32>, SparseVector)> {
        let dense = self.embed_dense(text)?;
        let sparse = self.sparse.embed_document(text);
        Ok((dense, sparse))
    }

    /// Embed a query (to search). Sparse uses BM25 query (unit) weighting.
    pub fn embed_query(&self, text: &str) -> Result<(Vec<f32>, SparseVector)> {
        let dense = self.embed_dense(text)?;
        let sparse = self.sparse.embed_query(text);
        Ok((dense, sparse))
    }
}
