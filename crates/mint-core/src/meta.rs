//! Side metadata that changes without re-embedding: which memories are archived
//! (by decay) and which summary node belongs to which entity. Persisted as
//! meta.json in the data dir.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const FILE: &str = "meta.json";

#[derive(Default, Serialize, Deserialize)]
struct MetaData {
    #[serde(default)]
    archived: HashSet<String>,
    /// entity id -> its consolidated summary memory id.
    #[serde(default)]
    summaries: HashMap<String, String>,
    /// memory id -> how many times it has been retrieved (feeds salience).
    #[serde(default)]
    access: HashMap<String, u32>,
    /// deleted memory id -> RFC3339 deletion time (for delete propagation).
    #[serde(default)]
    tombstones: HashMap<String, String>,
}

pub struct MetaStore {
    path: PathBuf,
    data: MetaData,
}

impl MetaStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(FILE);
        let data = if path.exists() {
            let bytes = std::fs::read(&path).context("read meta.json")?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            MetaData::default()
        };
        Ok(Self { path, data })
    }

    fn save(&self) -> Result<()> {
        std::fs::write(&self.path, serde_json::to_vec(&self.data)?).context("write meta.json")?;
        Ok(())
    }

    pub fn is_archived(&self, id: &str) -> bool {
        self.data.archived.contains(id)
    }

    pub fn archived_ids(&self) -> &HashSet<String> {
        &self.data.archived
    }

    pub fn archive(&mut self, id: &str) -> Result<()> {
        self.data.archived.insert(id.to_string());
        self.save()
    }

    pub fn clear_archive(&mut self) -> Result<usize> {
        let n = self.data.archived.len();
        self.data.archived.clear();
        self.save()?;
        Ok(n)
    }

    pub fn summary_for(&self, entity_id: &str) -> Option<String> {
        self.data.summaries.get(entity_id).cloned()
    }

    pub fn set_summary(&mut self, entity_id: &str, summary_id: &str) -> Result<()> {
        self.data
            .summaries
            .insert(entity_id.to_string(), summary_id.to_string());
        self.save()
    }

    /// Drop archived entries whose ids no longer exist.
    pub fn retain_existing(&mut self, existing: &HashSet<String>) -> Result<()> {
        self.data.archived.retain(|id| existing.contains(id));
        self.save()
    }

    // ---- access counts ----
    pub fn access_count(&self, id: &str) -> u32 {
        self.data.access.get(id).copied().unwrap_or(0)
    }
    pub fn access_snapshot(&self) -> HashMap<String, u32> {
        self.data.access.clone()
    }
    pub fn bump_access(&mut self, ids: &[String]) -> Result<()> {
        for id in ids {
            *self.data.access.entry(id.clone()).or_insert(0) += 1;
        }
        self.save()
    }

    // ---- tombstones (delete propagation) ----
    pub fn add_tombstone(&mut self, id: &str, ts: &str) -> Result<()> {
        self.data.tombstones.insert(id.to_string(), ts.to_string());
        self.data.archived.remove(id);
        self.data.access.remove(id);
        self.save()
    }
    pub fn is_tombstoned(&self, id: &str) -> bool {
        self.data.tombstones.contains_key(id)
    }
    pub fn tombstones(&self) -> &HashMap<String, String> {
        &self.data.tombstones
    }
}
