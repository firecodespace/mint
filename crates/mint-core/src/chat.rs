//! Chat + memory-capture logic, reusable across surfaces (Tauri app, CLI later).
//!
//! The orchestration (streaming to the UI, emitting stage events) lives in the
//! host; this module provides the primitives: building the RAG prompt from
//! retrieved memories, and extracting durable memories from a conversation turn.

use anyhow::Result;
use serde::Deserialize;

use crate::ollama::Ollama;
use crate::record::{MemoryKind, MemorySource, NewMemory, SearchResult, Sensitivity};

/// System persona + instructions, with the retrieved-memory context inlined.
pub fn system_prompt(context: &str) -> String {
    let base = "You are Mint, a local-first memory assistant that runs entirely on the \
user's device. You help the user think and remember. Be concise and direct. \
When the user's own memories below are relevant, ground your answer in them and \
refer to them naturally. If they are not relevant, just answer normally. Never \
invent memories the user did not provide.";

    if context.trim().is_empty() {
        format!("{base}\n\n[The user has no stored memories relevant to this message.]")
    } else {
        format!("{base}\n\nRelevant memories retrieved from the user's device:\n{context}")
    }
}

/// Format retrieved memories into a compact context block for the prompt.
pub fn format_context(results: &[SearchResult]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        let m = &r.memory;
        let text = truncate(&m.text, 400);
        out.push_str(&format!("{}. [{:?}] {}", i + 1, m.kind, m.title));
        if !m.title.is_empty() {
            out.push_str(": ");
        }
        out.push_str(&text);
        if !m.tags.is_empty() {
            out.push_str(&format!(" (tags: {})", m.tags.join(", ")));
        }
        out.push('\n');
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}...")
    }
}

#[derive(Debug, Deserialize)]
struct ExtractedItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractionResult {
    #[serde(default)]
    memories: Vec<ExtractedItem>,
}

const EXTRACT_SYSTEM: &str = "You extract durable, reusable memories from a conversation \
for a personal memory system. Return ONLY JSON of the form \
{\"memories\":[{\"title\":\"short label\",\"text\":\"the fact worth remembering\",\
\"kind\":\"note|observation|event|measurement\",\"tags\":[\"...\"]}]}. \
Include only information worth remembering long-term: facts about the user, decisions, \
preferences, entities, commitments, deadlines. Ignore small talk, questions, and \
transient chatter. If nothing is worth saving, return {\"memories\":[]}.";

/// Ask a fast local model to extract durable memories from one turn.
/// Returns ready-to-store `NewMemory` values (source = Chat).
pub fn extract_memories(
    ollama: &Ollama,
    model: &str,
    user_message: &str,
    assistant_message: &str,
) -> Result<Vec<NewMemory>> {
    let prompt = format!(
        "Conversation turn:\nUser: {user_message}\nAssistant: {assistant_message}\n\n\
Extract the durable memories as JSON."
    );
    let raw = ollama.generate(model, Some(EXTRACT_SYSTEM), &prompt, true)?;
    let parsed: ExtractionResult = serde_json::from_str(&raw).unwrap_or(ExtractionResult {
        memories: Vec::new(),
    });

    let memories = parsed
        .memories
        .into_iter()
        .filter(|it| !it.text.trim().is_empty())
        .map(|it| NewMemory {
            kind: MemoryKind::parse_lenient(&it.kind),
            title: if it.title.trim().is_empty() {
                truncate(&it.text, 60)
            } else {
                it.title
            },
            text: it.text,
            site_id: String::new(),
            asset_id: String::new(),
            geo: None,
            tags: it.tags,
            source: MemorySource::Chat,
            sensitivity: Sensitivity::Shareable,
        })
        .collect();
    Ok(memories)
}
