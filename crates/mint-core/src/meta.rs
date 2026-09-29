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
    /// topic id -> rolling-summary bookkeeping.
    #[serde(default)]
    topics: HashMap<String, TopicState>,
    /// This device's stable identity (edge <-> cloud provenance).
    #[serde(default)]
    device_id: String,
    /// memory id -> updated_at at the last successful sync (the 3-way merge
    /// base: tells "only one side changed" apart from a real conflict).
    #[serde(default)]
    synced: HashMap<String, String>,
    /// memory id -> user's explicit sync choice (true = may sync, false = keep
    /// on this device). Overrides the automatic policy.
    #[serde(default)]
    policy_overrides: HashMap<String, bool>,
}

/// Bookkeeping for a topic's rolling summary and naming.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopicState {
    /// Members added/moved since the summary was last refreshed.
    #[serde(default)]
    pub changes_since_summary: u32,
    /// RFC3339 time of the last summary refresh (None = never summarized).
    #[serde(default)]
    pub summarized_at: Option<String>,
    /// The user named this topic; automatic renaming must not override it.
    #[serde(default)]
    pub user_named: bool,
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

    // ---- topic bookkeeping (rolling summaries) ----
    pub fn topic_state(&self, id: &str) -> TopicState {
        self.data.topics.get(id).cloned().unwrap_or_default()
    }
    pub fn mark_topic_changed(&mut self, id: &str) -> Result<()> {
        self.data.topics.entry(id.to_string()).or_default().changes_since_summary += 1;
        self.save()
    }
    pub fn mark_topic_summarized(&mut self, id: &str, ts: &str) -> Result<()> {
        let s = self.data.topics.entry(id.to_string()).or_default();
        s.changes_since_summary = 0;
        s.summarized_at = Some(ts.to_string());
        self.save()
    }
    pub fn set_topic_user_named(&mut self, id: &str) -> Result<()> {
        self.data.topics.entry(id.to_string()).or_default().user_named = true;
        self.save()
    }
    pub fn remove_topic(&mut self, id: &str) -> Result<()> {
        self.data.topics.remove(id);
        self.save()
    }

    // ---- device identity ----
    /// This device's id, created once and persisted.
    pub fn device_id(&mut self) -> String {
        if self.data.device_id.is_empty() {
            self.data.device_id = format!("dev-{}", ulid::Ulid::generate());
            let _ = self.save();
        }
        self.data.device_id.clone()
    }

    // ---- sync merge base ----
    pub fn synced_version(&self, id: &str) -> Option<String> {
        self.data.synced.get(id).cloned()
    }
    pub fn set_synced_versions(&mut self, entries: &[(String, String)]) -> Result<()> {
        for (id, ts) in entries {
            self.data.synced.insert(id.clone(), ts.clone());
        }
        self.save()
    }

    // ---- user policy overrides ----
    pub fn policy_override(&self, id: &str) -> Option<bool> {
        self.data.policy_overrides.get(id).copied()
    }
    pub fn set_policy_override(&mut self, id: &str, share: bool) -> Result<()> {
        self.data.policy_overrides.insert(id.to_string(), share);
        self.save()
    }
}
