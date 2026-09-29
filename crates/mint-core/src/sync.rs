//! Edge -> cloud synchronization against a Qdrant Server over its REST API.
//! Uses ureq (already in tree). The cloud is optional: everything works offline,
//! and sync only runs when a server is reachable and the user is "online".

use std::io::Read;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Server collection that mirrors the local shard.
pub const COLLECTION: &str = "mint_memories";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
        }
    }
}

/// A point to upsert to the server: vectors + the memory payload.
pub struct ServerPoint {
    pub id: String, // uuid string
    pub dense: Vec<f32>,
    pub sparse_indices: Vec<u32>,
    pub sparse_values: Vec<f32>,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SyncReport {
    pub pushed: usize,
    pub pulled: usize,
    pub conflicts: usize,
}

pub struct SyncClient {
    cfg: SyncConfig,
}

impl SyncClient {
    pub fn new(cfg: SyncConfig) -> Self {
        Self { cfg }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.url.trim_end_matches('/'), path)
    }

    fn get(&self, path: &str) -> Result<String> {
        let mut req = ureq::get(self.url(path));
        if let Some(k) = &self.cfg.api_key {
            req = req.header("api-key", k);
        }
        let mut resp = req.call().map_err(|e| anyhow!("GET {path}: {e}"))?;
        let mut body = String::new();
        resp.body_mut().as_reader().read_to_string(&mut body)?;
        Ok(body)
    }

    fn send_json(&self, method: &str, path: &str, body: &serde_json::Value) -> Result<String> {
        let url = self.url(path);
        let builder = match method {
            "PUT" => ureq::put(url),
            "POST" => ureq::post(url),
            other => return Err(anyhow!("unsupported method {other}")),
        };
        let mut builder = builder.header("Content-Type", "application/json");
        if let Some(k) = &self.cfg.api_key {
            builder = builder.header("api-key", k);
        }
        let payload = serde_json::to_string(body)?;
        let resp = builder
            .send(payload)
            .map_err(|e| anyhow!("{method} {path}: {e}"))?;
        let mut out = String::new();
        resp.into_body().into_reader().read_to_string(&mut out)?;
        Ok(out)
    }

    /// Is the server reachable?
    pub fn reachable(&self) -> bool {
        self.get("/").is_ok()
    }

    /// Create the collection if it does not exist (dense + sparse vectors).
    pub fn ensure_collection(&self, dim: usize) -> Result<()> {
        if self.get(&format!("/collections/{COLLECTION}")).is_ok() {
            return Ok(());
        }
        let body = serde_json::json!({
            "vectors": { "dense": { "size": dim, "distance": "Cosine" } },
            "sparse_vectors": { "sparse": {} }
        });
        self.send_json("PUT", &format!("/collections/{COLLECTION}"), &body)?;
        Ok(())
    }

    pub fn upsert(&self, points: &[ServerPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let json_points: Vec<serde_json::Value> = points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "vector": {
                        "dense": p.dense,
                        "sparse": { "indices": p.sparse_indices, "values": p.sparse_values }
                    },
                    "payload": p.payload
                })
            })
            .collect();
        let body = serde_json::json!({ "points": json_points });
        self.send_json(
            "PUT",
            &format!("/collections/{COLLECTION}/points?wait=true"),
            &body,
        )?;
        Ok(())
    }

    /// Scroll every point's payload from the server.
    pub fn scroll_all(&self) -> Result<Vec<serde_json::Value>> {
        let mut payloads = Vec::new();
        let mut offset: Option<serde_json::Value> = None;
        loop {
            let mut body = serde_json::json!({
                "limit": 256, "with_payload": true, "with_vector": false
            });
            if let Some(o) = &offset {
                body["offset"] = o.clone();
            }
            let raw = self.send_json(
                "POST",
                &format!("/collections/{COLLECTION}/points/scroll"),
                &body,
            )?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            let result = &v["result"];
            if let Some(points) = result["points"].as_array() {
                for p in points {
                    payloads.push(p["payload"].clone());
                }
            }
            match result.get("next_page_offset") {
                Some(next) if !next.is_null() => offset = Some(next.clone()),
                _ => break,
            }
        }
        Ok(payloads)
    }
}
