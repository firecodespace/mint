//! Persistent relationship layer over memories. Edges are established by the AI
//! (embedding nearest-neighbours = "related") and by document structure
//! (chunk -> document = "part_of"). Stored as edges.json in the data dir.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const FILE: &str = "edges.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relation: String, // "related" | "part_of"
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Default)]
pub struct GraphStore {
    path: PathBuf,
    edges: Vec<GraphEdge>,
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
        Ok(Self { path, edges })
    }

    fn save(&self) -> Result<()> {
        let bytes = serde_json::to_vec(&self.edges)?;
        std::fs::write(&self.path, bytes).context("write edges.json")?;
        Ok(())
    }

    pub fn all(&self) -> &[GraphEdge] {
        &self.edges
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

    pub fn add_part_of(&mut self, chunk: &str, document: &str) -> Result<()> {
        self.edges.push(GraphEdge {
            from: chunk.to_string(),
            to: document.to_string(),
            relation: "part_of".into(),
        });
        self.save()
    }

    /// Remove every edge touching `id` (on node deletion).
    pub fn remove_node(&mut self, id: &str) -> Result<()> {
        self.edges.retain(|e| e.from != id && e.to != id);
        self.save()
    }
}
