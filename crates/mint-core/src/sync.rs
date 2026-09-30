// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
//! Edge -> cloud synchronization against a Qdrant Server over its REST API.
//! Uses ureq (already in tree). The cloud is optional: everything works offline,
//! and sync only runs when a server is reachable and the user is "online".

use std::collections::BTreeMap;
use std::io::Read;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Default server collection that mirrors the local shard.
pub const COLLECTION: &str = "mint_memories";

/// Timeout for sync transfers (push/pull/scroll).
const SYNC_TIMEOUT: Duration = Duration::from_secs(15);
/// Timeout for latency-sensitive calls (reachability, cloud search in chat).
const FAST_TIMEOUT: Duration = Duration::from_millis(1500);

fn default_collection() -> String {
    COLLECTION.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Server collection (tests use throwaway ones; the app uses the default).
    #[serde(default = "default_collection")]
    pub collection: String,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            collection: default_collection(),
        }
    }
}

/// Server collection that records deletions for cross-device propagation.
pub const TOMBSTONES: &str = "mint_tombstones";

/// A point to upsert to the server: vectors + the memory payload.
pub struct ServerPoint {
    pub id: String, // uuid string
    pub dense: Vec<f32>,
    pub sparse_indices: Vec<u32>,
    pub sparse_values: Vec<f32>,
    pub payload: serde_json::Value,
}

/// A tombstone point (a deleted memory).
pub struct TombstonePoint {
    pub id_uuid: String,
    pub memory_id: String,
    pub deleted_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyncReport {
    pub pushed: usize,
    pub pulled: usize,
    /// Edits made on BOTH sides since the last sync (the losing edit is kept
    /// as an older version, never dropped).
    pub conflicts: usize,
    /// Memories the policy kept on this device (not pushed).
    #[serde(default)]
    pub withheld: usize,
    /// Why they were kept local (category -> count).
    #[serde(default)]
    pub withheld_by: BTreeMap<String, usize>,
    /// Cloud copies removed because the memory is now private.
    #[serde(default)]
    pub retracted: usize,
    /// Other devices' raw memories left in the cloud (tiered pull); still
    /// reachable through cloud search when online.
    #[serde(default)]
    pub cloud_only: usize,
}

pub struct SyncClient {
    cfg: SyncConfig,
    agent: ureq::Agent,
    fast: ureq::Agent,
}

fn agent_with_timeout(t: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(t))
        .build()
        .into()
}

impl SyncClient {
    pub fn new(cfg: SyncConfig) -> Self {
        Self {
            cfg,
            agent: agent_with_timeout(SYNC_TIMEOUT),
            fast: agent_with_timeout(FAST_TIMEOUT),
        }
    }

    fn collection(&self) -> &str {
        &self.cfg.collection
    }

    fn tombstones(&self) -> String {
        if self.cfg.collection == COLLECTION {
            TOMBSTONES.to_string()
        } else {
            format!("{}_tombstones", self.cfg.collection)
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.url.trim_end_matches('/'), path)
    }

    fn get(&self, path: &str) -> Result<String> {
        self.get_with(&self.agent, path)
    }

    fn get_with(&self, agent: &ureq::Agent, path: &str) -> Result<String> {
        let mut req = agent.get(self.url(path));
        if let Some(k) = &self.cfg.api_key {
            req = req.header("api-key", k);
        }
        let mut resp = req.call().map_err(|e| anyhow!("GET {path}: {e}"))?;
        let mut body = String::new();
        resp.body_mut().as_reader().read_to_string(&mut body)?;
        Ok(body)
    }

    fn send_json(&self, method: &str, path: &str, body: &serde_json::Value) -> Result<String> {
        self.send_json_with(&self.agent, method, path, body)
    }

    fn send_json_with(
        &self,
        agent: &ureq::Agent,
        method: &str,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<String> {
        let url = self.url(path);
        let builder = match method {
            "PUT" => agent.put(url),
            "POST" => agent.post(url),
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

    /// Is the server reachable? (short timeout)
    pub fn reachable(&self) -> bool {
        self.get_with(&self.fast, "/").is_ok()
    }

    /// Cloud search: hybrid (dense + sparse, RRF) query against the server,
    /// short timeout so chat never stalls on the network. Returns
    /// (payload, score) pairs.
    pub fn query(
        &self,
        dense: &[f32],
        sparse_indices: &[u32],
        sparse_values: &[f32],
        limit: usize,
    ) -> Result<Vec<(serde_json::Value, f32)>> {
        let body = serde_json::json!({
            "prefetch": [
                { "query": dense, "using": "dense", "limit": 30 },
                { "query": { "indices": sparse_indices, "values": sparse_values },
                  "using": "sparse", "limit": 30 }
            ],
            "query": { "fusion": "rrf" },
            "limit": limit,
            "with_payload": true
        });
        let raw = self.send_json_with(
            &self.fast,
            "POST",
            &format!("/collections/{}/points/query", self.collection()),
            &body,
        )?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let points = v["result"]["points"]
            .as_array()
            .or_else(|| v["result"].as_array())
            .cloned()
            .unwrap_or_default();
        Ok(points
            .into_iter()
            .map(|p| (p["payload"].clone(), p["score"].as_f64().unwrap_or(0.0) as f32))
            .collect())
    }

    /// Drop this client's collections (test cleanup only).
    pub fn delete_collections(&self) -> Result<()> {
        for c in [self.collection().to_string(), self.tombstones()] {
            let mut req = self.agent.delete(self.url(&format!("/collections/{c}")));
            if let Some(k) = &self.cfg.api_key {
                req = req.header("api-key", k);
            }
            let _ = req.call();
        }
        Ok(())
    }

    /// Create the collection if it does not exist (dense + sparse vectors).
    pub fn ensure_collection(&self, dim: usize) -> Result<()> {
        if self.get(&format!("/collections/{}", self.collection())).is_ok() {
            return Ok(());
        }
        let body = serde_json::json!({
            "vectors": { "dense": { "size": dim, "distance": "Cosine" } },
            "sparse_vectors": { "sparse": {} }
        });
        self.send_json("PUT", &format!("/collections/{}", self.collection()), &body)?;
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
            &format!("/collections/{}/points?wait=true", self.collection()),
            &body,
        )?;
        Ok(())
    }

    pub fn ensure_tombstone_collection(&self) -> Result<()> {
        if self.get(&format!("/collections/{}", self.tombstones())).is_ok() {
            return Ok(());
        }
        let body = serde_json::json!({ "vectors": { "t": { "size": 1, "distance": "Dot" } } });
        self.send_json("PUT", &format!("/collections/{}", self.tombstones()), &body)?;
        Ok(())
    }

    pub fn upsert_tombstones(&self, points: &[TombstonePoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let json_points: Vec<serde_json::Value> = points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id_uuid,
                    "vector": { "t": [0.0] },
                    "payload": { "id": p.memory_id, "deleted_at": p.deleted_at }
                })
            })
            .collect();
        self.send_json(
            "PUT",
            &format!("/collections/{}/points?wait=true", self.tombstones()),
            &serde_json::json!({ "points": json_points }),
        )?;
        Ok(())
    }

    /// (memory_id, deleted_at) for every tombstone on the server.
    pub fn scroll_tombstones(&self) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        let mut offset: Option<serde_json::Value> = None;
        loop {
            let mut body = serde_json::json!({ "limit": 256, "with_payload": true, "with_vector": false });
            if let Some(o) = &offset {
                body["offset"] = o.clone();
            }
            let raw = self.send_json(
                "POST",
                &format!("/collections/{}/points/scroll", self.tombstones()),
                &body,
            )?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            let result = &v["result"];
            if let Some(points) = result["points"].as_array() {
                for p in points {
                    let id = p["payload"]["id"].as_str().unwrap_or("").to_string();
                    let ts = p["payload"]["deleted_at"].as_str().unwrap_or("").to_string();
                    if !id.is_empty() {
                        out.push((id, ts));
                    }
                }
            }
            match result.get("next_page_offset") {
                Some(next) if !next.is_null() => offset = Some(next.clone()),
                _ => break,
            }
        }
        Ok(out)
    }

    /// Delete points from the main collection by uuid id.
    pub fn delete_points(&self, uuid_ids: &[String]) -> Result<()> {
        if uuid_ids.is_empty() {
            return Ok(());
        }
        self.send_json(
            "POST",
            &format!("/collections/{}/points/delete?wait=true", self.collection()),
            &serde_json::json!({ "points": uuid_ids }),
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
                &format!("/collections/{}/points/scroll", self.collection()),
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
