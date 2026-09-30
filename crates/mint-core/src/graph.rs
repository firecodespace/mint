// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
//! Persistent relationship layer over memories. Edges are established by the AI
//! (embedding nearest-neighbours = "related") and by document structure
//! (chunk -> document = "part_of"). Stored as edges.json in the data dir.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const FILE: &str = "edges.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relation: String, // "related" | "part_of" | "mentions" | "in_topic"
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub parent_id: Option<String>,
    /// Topic the node is filed under (chunks inherit their document's topic).
    pub topic_id: Option<String>,
    pub salience: f32,
    /// Replaced by a newer version (version chain).
    pub superseded: bool,
    /// Kept on this device by the sync policy.
    pub local_only: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// Memories decayed into the archive ("forgotten"), not drawn.
    pub archived: usize,
}

#[derive(Default)]
pub struct GraphStore {
    path: PathBuf,
    edges: Vec<GraphEdge>,
    /// Cached connectivity degree per node (edges touching it). Rebuilt on every
    /// mutation so retrieval can read degrees in O(1) instead of scanning edges.
    degree: HashMap<String, usize>,
}

impl GraphStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(FILE);
        let edges = if path.exists() {
            let bytes = std::fs::read(&path).context("read edges.json")?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut g = Self {
            path,
            edges,
            degree: HashMap::new(),
        };
        g.reindex();
        Ok(g)
    }

    fn reindex(&mut self) {
        self.degree.clear();
        for e in &self.edges {
            *self.degree.entry(e.from.clone()).or_insert(0) += 1;
            *self.degree.entry(e.to.clone()).or_insert(0) += 1;
        }
    }

    fn save(&mut self) -> Result<()> {
        self.reindex();
        let bytes = serde_json::to_vec(&self.edges)?;
        std::fs::write(&self.path, bytes).context("write edges.json")?;
        Ok(())
    }

    pub fn all(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Connectivity degree of one node (O(1)).
    pub fn degree(&self, id: &str) -> usize {
        self.degree.get(id).copied().unwrap_or(0)
    }

    /// Snapshot of all degrees (for full listings).
    pub fn degrees(&self) -> HashMap<String, usize> {
        self.degree.clone()
    }

    /// Replace the "related" edges originating from `from` with edges to `tos`.
    pub fn set_related(&mut self, from: &str, tos: &[String]) -> Result<()> {
        self.edges
            .retain(|e| !(e.relation == "related" && e.from == from));
        for to in tos {
            if to != from {
                self.edges.push(GraphEdge {
                    from: from.to_string(),
                    to: to.clone(),
                    relation: "related".into(),
                });
            }
        }
        self.save()
    }

    /// Replace the "mentions" edges from `from` with edges to the given entities.
    pub fn set_mentions(&mut self, from: &str, entity_ids: &[String]) -> Result<()> {
        self.edges
            .retain(|e| !(e.relation == "mentions" && e.from == from));
        for to in entity_ids {
            if to != from {
                self.edges.push(GraphEdge {
                    from: from.to_string(),
                    to: to.clone(),
                    relation: "mentions".into(),
                });
            }
        }
        self.save()
    }

    pub fn add_part_of(&mut self, chunk: &str, document: &str) -> Result<()> {
        self.add_part_of_many(&[chunk.to_string()], document)
    }

    /// Link many chunks to their document with a single write (ingestion).
    pub fn add_part_of_many(&mut self, chunks: &[String], document: &str) -> Result<()> {
        for c in chunks {
            self.edges.push(GraphEdge {
                from: c.clone(),
                to: document.to_string(),
                relation: "part_of".into(),
            });
        }
        self.save()
    }

    /// Add one edge unless an identical edge already exists.
    pub fn add_edge(&mut self, from: &str, to: &str, relation: &str) -> Result<()> {
        if from == to
            || self
                .edges
                .iter()
                .any(|e| e.from == from && e.to == to && e.relation == relation)
        {
            return Ok(());
        }
        self.edges.push(GraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
        });
        self.save()
    }

    /// Point every in_topic edge that targets `from` at `into` (topic merge).
    pub fn retarget_topic(&mut self, from: &str, into: &str) -> Result<()> {
        for e in self.edges.iter_mut() {
            if e.relation == "in_topic" && e.to == from {
                e.to = into.to_string();
            }
        }
        self.edges.retain(|e| !(e.relation == "in_topic" && e.from == e.to));
        self.save()
    }

    /// Record that a memory/document is filed under a topic. Replaces any prior
    /// in_topic edge from `member` so reassignment stays clean (one topic each).
    pub fn add_in_topic(&mut self, member: &str, topic: &str) -> Result<()> {
        self.edges
            .retain(|e| !(e.relation == "in_topic" && e.from == member));
        if member != topic {
            self.edges.push(GraphEdge {
                from: member.to_string(),
                to: topic.to_string(),
                relation: "in_topic".into(),
            });
        }
        self.save()
    }

    /// Remove every edge touching `id` (on node deletion).
    pub fn remove_node(&mut self, id: &str) -> Result<()> {
        self.edges.retain(|e| e.from != id && e.to != id);
        self.save()
    }
}
