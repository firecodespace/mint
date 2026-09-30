// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
//! Chat + memory-capture logic, reusable across surfaces (Tauri app, CLI later).
//!
//! The orchestration (streaming to the UI, emitting stage events) lives in the
//! host; this module provides the primitives: building the RAG prompt from
//! retrieved memories, and extracting durable memories from a conversation turn.

use anyhow::Result;
use serde_json::Value;

use crate::ollama::Ollama;
use crate::record::{MemoryKind, MemorySource, NewMemory, SearchResult, Sensitivity};

/// Parse model output as JSON, tolerating prose around it: the whole string,
/// else the outermost {...} or [...] span. Small local models often wrap or
/// decorate JSON even in JSON mode.
pub fn parse_json_lenient(raw: &str) -> Option<Value> {
    let t = raw.trim();
    if let Ok(v) = serde_json::from_str(t) {
        return Some(v);
    }
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(a), Some(b)) = (t.find(open), t.rfind(close)) {
            if b > a {
                if let Ok(v) = serde_json::from_str(&t[a..=b]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// The array under `key` (any key casing), or the value itself if it is an
/// array. Missing/mistyped -> empty.
pub fn json_array_field(v: &Value, key: &str) -> Vec<Value> {
    if let Some(a) = v.as_array() {
        return a.clone();
    }
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if k.eq_ignore_ascii_case(key) {
                if let Some(a) = val.as_array() {
                    return a.clone();
                }
            }
        }
    }
    Vec::new()
}

/// A string field, or "" when missing / null / not a string.
fn str_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// A list of strings, skipping non-string entries.
fn str_list(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

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
        // Version chains: tell the model which facts are current.
        let status = if m.superseded_by.is_some() {
            " (OUTDATED: replaced by a newer memory)"
        } else if m.supersedes.is_some() {
            " (current: updates an earlier memory)"
        } else {
            ""
        };
        out.push_str(&format!("{}. [{:?}]{status} {}", i + 1, m.kind, m.title));
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

/// Memories from the user's OTHER devices, found through cloud search while
/// online (labeled so the model knows their provenance).
pub fn format_cloud_context(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nFrom the user's other devices (cloud):\n");
    for r in results {
        out.push_str(&format!("- {}\n", truncate(&r.memory.text, 400)));
    }
    out
}

/// Subject overviews (rolling topic summaries) for the subjects the question
/// belongs to. Placed before the individual memories so the model reads the
/// big picture first, then the details.
pub fn format_topic_overview(topics: &[(String, String)]) -> String {
    if topics.is_empty() {
        return String::new();
    }
    let mut out = String::from("Subject overview (distilled from the user's memories):\n");
    for (name, summary) in topics {
        out.push_str(&format!("- {name}: {}\n", truncate(summary, 700)));
    }
    out.push_str("\nIndividual memories:\n");
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

/// A message that is a question, request, command, or greeting yields no
/// memories. Only genuinely declarative statements should be captured; being
/// strict here keeps meta-turns ("check my resume", "what are my strengths")
/// from manufacturing junk notes that later pollute retrieval.
pub fn is_query_only(msg: &str) -> bool {
    let m = msg.trim();
    if m.is_empty() {
        return true;
    }
    let words = m.split_whitespace().count();
    // Any question mark on a reasonably short message => a question.
    if m.ends_with('?') && words < 24 {
        return true;
    }
    let lower = m.to_lowercase();
    const GREETINGS: &[&str] = &["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"];
    if GREETINGS.iter().any(|g| lower == *g) {
        return true;
    }
    // First word decides intent: a question word or imperative verb means the
    // user is asking/instructing, not stating a durable fact about themselves.
    let first = lower
        .split(|c: char| !c.is_alphanumeric())
        .find(|w| !w.is_empty())
        .unwrap_or("");
    const QUESTION_WORDS: &[&str] = &[
        "what", "whats", "who", "whos", "when", "where", "why", "how", "hows", "which",
        "whose", "whom", "is", "are", "am", "do", "does", "did", "can", "could", "would",
        "should", "will", "was", "were", "may", "might", "have", "has", "had",
    ];
    const COMMAND_WORDS: &[&str] = &[
        "check", "tell", "show", "find", "give", "list", "explain", "summarize", "summarise",
        "describe", "look", "search", "get", "help", "remind", "please", "compare", "analyze",
        "analyse", "write", "make", "create", "generate", "suggest", "recommend", "define",
        "translate", "fix", "draft", "review", "calculate", "convert", "pull", "fetch", "open",
        "let", "lets",
        // Scheduling/admin commands: the timeline extractor handles their dates;
        // they must not ALSO become free-floating "Deadline" notes.
        "set", "schedule", "book", "cancel", "send", "remove", "delete", "rename",
    ];
    QUESTION_WORDS.contains(&first) || COMMAND_WORDS.contains(&first)
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

/// Parse an event-extraction reply leniently into timeline memories. Items
/// need a title and a VALID calendar date (YYYY-MM-DD); bad items are skipped
/// without discarding good ones.
pub fn parse_events(raw: &str) -> Vec<NewMemory> {
    let Some(v) = parse_json_lenient(raw) else {
        return Vec::new();
    };
    json_array_field(&v, "items")
        .iter()
        .filter_map(|it| {
            let title = str_field(it, "title");
            let due = str_field(it, "due");
            if title.is_empty() || chrono::NaiveDate::parse_from_str(&due, "%Y-%m-%d").is_err() {
                return None;
            }
            let kind_tag = if str_field(it, "kind") == "task" { "task" } else { "event" };
            Some(NewMemory {
                kind: MemoryKind::Event,
                text: format!("{title} — {kind_tag} on {due}"),
                title,
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags: vec![kind_tag.to_string()],
                source: MemorySource::Chat,
                sensitivity: Sensitivity::Shareable,
                parent_id: None,
                topic_id: None,
                due_at: Some(due),
            })
        })
        .collect()
}

/// Parse a memory-extraction reply leniently (junk facts dropped, at most 3).
pub fn parse_memories(raw: &str) -> Vec<NewMemory> {
    let Some(v) = parse_json_lenient(raw) else {
        return Vec::new();
    };
    json_array_field(&v, "memories")
        .iter()
        .filter_map(|it| {
            let text = str_field(it, "text");
            if is_junk(&text) {
                return None;
            }
            let title = str_field(it, "title");
            Some(NewMemory {
                kind: MemoryKind::parse_lenient(&str_field(it, "kind")),
                title: if title.is_empty() { truncate(&text, 60) } else { title },
                text,
                site_id: String::new(),
                asset_id: String::new(),
                geo: None,
                tags: str_list(it, "tags"),
                source: MemorySource::Chat,
                sensitivity: Sensitivity::Shareable,
                parent_id: None,
                topic_id: None,
                due_at: None,
            })
        })
        .take(3) // don't over-capture from a single turn
        .collect()
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
    Ok(parse_events(&raw))
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
    Ok(parse_memories(&raw))
}

#[cfg(test)]
mod tests {
    use super::{is_query_only, parse_events, parse_memories};

    #[test]
    fn extraction_survives_messy_model_output() {
        // One item with a null field must not discard the valid one.
        let raw = r#"{"memories":[{"title":null,"text":"I prefer dark roast","kind":"note","tags":null},
                      {"title":"Bad","text":"unknown"}]}"#;
        let m = parse_memories(raw);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].text, "I prefer dark roast");
        assert!(!m[0].title.is_empty());
        // Prose around JSON, capitalized key.
        assert_eq!(parse_memories("Sure! {\"Memories\":[{\"text\":\"My exam is Friday\"}]}").len(), 1);
        assert!(parse_memories("no json here").is_empty());
    }

    #[test]
    fn events_need_a_real_date() {
        let raw = r#"{"items":[{"title":"Submit report","due":"2026-10-02","kind":"task"},
                     {"title":"Bad date","due":"Friday"},
                     {"title":"Impossible","due":"2026-13-45"},
                     {"title":null,"due":"2026-10-03"}]}"#;
        let e = parse_events(raw);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].due_at.as_deref(), Some("2026-10-02"));
        assert_eq!(e[0].tags, vec!["task"]);
    }

    #[test]
    fn statements_are_captured() {
        for s in [
            "My name is Alex",
            "I work at Brightline Labs as the founder",
            "I decided to use Rust for the backend",
            "My exam is on Friday",
            "We moved the launch to October",
            "Started learning piano this week",
        ] {
            assert!(!is_query_only(s), "should capture: {s}");
        }
    }

    #[test]
    fn questions_and_commands_are_not_captured() {
        for s in [
            "check from my resume",
            "what type of internships are best for me?",
            "tell me about my projects",
            "how does HCMA work",
            "summarize my exoplanet research",
            "hi",
            "thanks",
            "What's my name?",
            "explain OPT",
            "set a deadline for bringing our benchmarks for core-sum by the end of this week",
            "schedule a call with the DSO next Tuesday",
            "can you set a deadline for friday",
        ] {
            assert!(is_query_only(s), "should NOT capture: {s}");
        }
    }
}
