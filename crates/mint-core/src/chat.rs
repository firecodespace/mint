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

const EXTRACT_SYSTEM: &str = "You extract durable memories from ONLY the user's latest \
message, for a personal memory system. Return ONLY JSON: \
{\"memories\":[{\"title\":\"short label\",\"text\":\"the fact\",\
\"kind\":\"note|observation|event|measurement\",\"tags\":[\"...\"]}]}.\n\
STRICT RULES:\n\
- Extract only NEW facts the USER explicitly stated in their message (their name, \
preferences, decisions, plans, commitments, deadlines, facts about their world).\n\
- If the user's message is a QUESTION, a request, a greeting, or small talk, extract \
NOTHING.\n\
- Never extract the assistant's words. Never restate or duplicate something the user \
is merely asking about.\n\
- When in doubt, extract nothing. Most turns should yield an empty list.\n\
Return {\"memories\":[]} when there is nothing genuinely new to store.";

/// A message that is purely a question or greeting yields no memories.
fn is_query_only(msg: &str) -> bool {
    let m = msg.trim();
    if m.is_empty() {
        return true;
    }
    let words = m.split_whitespace().count();
    // Short messages ending in a question mark are questions, not statements.
    if m.ends_with('?') && words < 16 {
        return true;
    }
    let lower = m.to_lowercase();
    const GREETINGS: &[&str] = &["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"];
    GREETINGS.iter().any(|g| lower == *g)
}

/// Reject junk facts the model sometimes emits (empty / "unknown" / too short).
fn is_junk(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    if t.chars().count() < 3 {
        return true;
    }
    matches!(t.as_str(), "unknown" | "n/a" | "na" | "none" | "null" | "the user" | "user")
}

/// Today's date as YYYY-MM-DD, for resolving relative dates.
pub fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// Cheap guard: does the message plausibly reference a date/time/deadline?
/// Avoids an LLM call on turns that clearly have no scheduling content.
pub fn mentions_time(msg: &str) -> bool {
    let m = msg.to_lowercase();
    const HINTS: &[&str] = &[
        "today", "tomorrow", "tonight", "yesterday", "deadline", "due", "by ", "on ",
        "next ", "this ", "week", "month", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december",
        "am ", "pm", "o'clock", "schedule", "remind", "meeting", "appointment", "submit",
        "apply", "exam", "quarter",
    ];
    if HINTS.iter().any(|h| m.contains(h)) {
        return true;
    }
    // A date-like number pattern (e.g. 12/5, 2026-03-01, "15th").
    m.chars().any(|c| c.is_ascii_digit())
        && (m.contains('/') || m.contains('-') || m.contains("th") || m.contains("st")
            || m.contains("nd") || m.contains("rd"))
}

#[derive(Debug, Deserialize)]
struct EventItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    due: String,
    #[serde(default)]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct EventResult {
    #[serde(default)]
    items: Vec<EventItem>,
}

/// Extract tasks/deadlines/dated events from the user's message, resolving
/// relative dates against `today` (YYYY-MM-DD). Returns ready-to-store memories.
pub fn extract_events(
    ollama: &Ollama,
    model: &str,
    user_message: &str,
    today: &str,
) -> Result<Vec<NewMemory>> {
    if !mentions_time(user_message) {
        return Ok(Vec::new());
    }
    let system = format!(
        "You extract tasks, deadlines, and dated events from the user's message. \
Today's date is {today}. Resolve relative dates (tomorrow, Friday, next week) to an \
absolute calendar date. Return ONLY JSON of the form \
{{\"items\":[{{\"title\":\"short label\",\"due\":\"YYYY-MM-DD\",\"kind\":\"task|event\"}}]}}. \
Only include items that have a real date or deadline. If none, return {{\"items\":[]}}."
    );
    let raw = ollama.generate(model, Some(&system), user_message, true)?;
    let parsed: EventResult = serde_json::from_str(&raw).unwrap_or(EventResult { items: Vec::new() });

    let items = parsed
        .items
        .into_iter()
        .filter(|it| it.due.trim().len() >= 8 && !it.title.trim().is_empty())
        .map(|it| {
            let kind_tag = if it.kind == "task" { "task" } else { "event" };
            NewMemory {
                kind: MemoryKind::Event,
                text: format!("{} — {} on {}", it.title, kind_tag, it.due.trim()),
                title: it.title,
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags: vec![kind_tag.to_string()],
                source: MemorySource::Chat,
                sensitivity: Sensitivity::Shareable,
                parent_id: None,
                due_at: Some(it.due.trim().to_string()),
            }
        })
        .collect();
    Ok(items)
}

/// Ask a fast local model to extract durable memories from one turn.
/// Returns ready-to-store `NewMemory` values (source = Chat).
pub fn extract_memories(
    ollama: &Ollama,
    model: &str,
    user_message: &str,
    assistant_message: &str,
) -> Result<Vec<NewMemory>> {
    // Assistant text is context only; extraction targets the user's new facts.
    let _ = assistant_message;
    // Cheap guard before spending an LLM call: questions/greetings store nothing.
    if is_query_only(user_message) {
        return Ok(Vec::new());
    }
    let prompt = format!(
        "The user's latest message:\n\"{user_message}\"\n\n\
Extract only NEW durable facts the user stated. If it is a question or small talk, \
return {{\"memories\":[]}}."
    );
    let raw = ollama.generate(model, Some(EXTRACT_SYSTEM), &prompt, true)?;
    let parsed: ExtractionResult = serde_json::from_str(&raw).unwrap_or(ExtractionResult {
        memories: Vec::new(),
    });

    let memories = parsed
        .memories
        .into_iter()
        .filter(|it| !is_junk(&it.text))
        .take(3) // don't over-capture from a single turn
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
            parent_id: None,
            due_at: None,
        })
        .collect();
    Ok(memories)
}
