// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
//! Local conversation store. Persists chat threads as JSON in the data dir so
//! "continue chat" survives restarts. Each conversation links the memory ids it
//! produced, so deleting a chat can optionally purge those memories too.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const FILE: &str = "conversations.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub ts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub messages: Vec<StoredMessage>,
    /// Memory ids captured during this conversation.
    #[serde(default)]
    pub memory_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub updated_at: String,
    pub message_count: usize,
    pub memory_count: usize,
}

pub struct ConversationStore {
    path: PathBuf,
    convos: Vec<Conversation>,
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

impl ConversationStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(FILE);
        let convos = if path.exists() {
            let bytes = std::fs::read(&path).context("read conversations.json")?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Self { path, convos })
    }

    fn save(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.convos)?;
        std::fs::write(&self.path, bytes).context("write conversations.json")?;
        Ok(())
    }

    pub fn list(&self) -> Vec<ConversationSummary> {
        let mut out: Vec<ConversationSummary> = self
            .convos
            .iter()
            .map(|c| ConversationSummary {
                id: c.id.clone(),
                title: c.title.clone(),
                updated_at: c.updated_at.clone(),
                message_count: c.messages.len(),
                memory_count: c.memory_ids.len(),
            })
            .collect();
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }

    pub fn get(&self, id: &str) -> Option<Conversation> {
        self.convos.iter().find(|c| c.id == id).cloned()
    }

    pub fn create(&mut self, title: Option<String>) -> Result<Conversation> {
        let ts = now();
        let convo = Conversation {
            id: ulid::Ulid::generate().to_string(),
            title: title.unwrap_or_default(),
            created_at: ts.clone(),
            updated_at: ts,
            messages: Vec::new(),
            memory_ids: Vec::new(),
        };
        self.convos.push(convo.clone());
        self.save()?;
        Ok(convo)
    }

    pub fn rename(&mut self, id: &str, title: String) -> Result<()> {
        if let Some(c) = self.convos.iter_mut().find(|c| c.id == id) {
            c.title = title;
            c.updated_at = now();
        }
        self.save()
    }

    /// Append a completed turn (user + assistant) and link captured memory ids.
    /// Sets the title from the first user message when still empty.
    pub fn append_turn(
        &mut self,
        id: &str,
        user: &str,
        assistant: &str,
        memory_ids: &[String],
    ) -> Result<()> {
        if let Some(c) = self.convos.iter_mut().find(|c| c.id == id) {
            let ts = now();
            if c.title.trim().is_empty() {
                c.title = user.chars().take(48).collect::<String>();
            }
            c.messages.push(StoredMessage {
                role: "user".into(),
                content: user.to_string(),
                ts: ts.clone(),
            });
            c.messages.push(StoredMessage {
                role: "assistant".into(),
                content: assistant.to_string(),
                ts: ts.clone(),
            });
            c.memory_ids.extend_from_slice(memory_ids);
            c.updated_at = ts;
        }
        self.save()
    }

    /// Remove a conversation. Returns its linked memory ids (for optional purge).
    pub fn delete(&mut self, id: &str) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        if let Some(pos) = self.convos.iter().position(|c| c.id == id) {
            ids = self.convos[pos].memory_ids.clone();
            self.convos.remove(pos);
            self.save()?;
        }
        Ok(ids)
    }
}
