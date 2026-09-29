//! Entity extraction: pull the key named entities (typed) from text so they can
//! become connective hub nodes in the knowledge graph. Local (Ollama).

use anyhow::Result;
use serde::Deserialize;

use crate::ollama::Ollama;

const SYSTEM: &str = "Extract the key named entities from the text for a knowledge graph. \
Return ONLY JSON of the form {\"entities\":[{\"name\":\"Canonical Name\",\"type\":\"person|org|place|project|product|concept\"}]}. \
Rules: use short canonical names; at most 8; skip generic words, pronouns, dates, and filler. \
If there are no meaningful entities, return {\"entities\":[]}.";

#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedEntity {
    #[serde(default)]
    pub name: String,
    #[serde(default, rename = "type")]
    pub etype: String,
}

#[derive(Debug, Deserialize)]
struct EntityResult {
    #[serde(default)]
    entities: Vec<ExtractedEntity>,
}

/// A normalized key for deduping entity names: lowercased, punctuation-stripped,
/// with common org suffixes removed ("Stanford University" ~ "Stanford").
pub fn normalize(name: &str) -> String {
    let lower = name.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();
    const SUFFIXES: &[&str] = &[
        "university", "univ", "college", "institute", "inc", "llc", "ltd", "corp",
        "corporation", "company", "co", "the",
    ];
    let words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|w| !SUFFIXES.contains(w))
        .collect();
    let joined = words.join(" ");
    if joined.trim().is_empty() {
        lower.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        joined
    }
}

/// True for names that are dates, relative-time phrases, quantities, or generic
/// filler — these must never become entity hubs (they clutter the graph and add
/// no connective value). The model is told to skip them but is unreliable, so we
/// enforce it deterministically.
fn is_temporal_or_filler(name: &str) -> bool {
    let n = name.trim().to_lowercase();
    if n.is_empty() {
        return true;
    }
    const MONTHS: &[&str] = &[
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    ];
    const DAYS: &[&str] = &[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ];
    const RELATIVE: &[&str] = &[
        "today", "tomorrow", "yesterday", "tonight", "now", "soon", "later", "week", "month",
        "year", "day", "days", "weeks", "months", "years", "weekend", "morning", "evening",
        "afternoon", "this week", "next week", "last week", "this month", "next month",
        "this year", "next year", "end of year", "end of week", "end of month", "deadline",
        "date", "time",
    ];
    const FILLER: &[&str] = &[
        "thing", "things", "stuff", "goal", "goals", "task", "tasks", "note", "notes",
        "idea", "ideas", "user", "me", "you", "it", "they", "this", "that", "project",
        "project deadline approach", "deadline approach",
    ];
    if MONTHS.contains(&n.as_str())
        || DAYS.contains(&n.as_str())
        || RELATIVE.contains(&n.as_str())
        || FILLER.contains(&n.as_str())
    {
        return true;
    }
    // Pure numbers / dates / times ("2026", "12/5", "9:30").
    if n.chars().all(|c| c.is_ascii_digit() || matches!(c, '/' | '-' | ':' | '.' | ' ')) {
        return true;
    }
    // Leading month or weekday ("January 2026", "Friday afternoon").
    let first = n.split_whitespace().next().unwrap_or("");
    MONTHS.contains(&first) || DAYS.contains(&first)
}

/// Extract up to 8 typed entities from `text`.
pub fn extract(ollama: &Ollama, model: &str, text: &str) -> Result<Vec<ExtractedEntity>> {
    let sample: String = text.chars().take(3000).collect();
    let raw = ollama.generate(model, Some(SYSTEM), &sample, true)?;
    let parsed: EntityResult = serde_json::from_str(&raw).unwrap_or(EntityResult {
        entities: Vec::new(),
    });

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for e in parsed.entities {
        let name = e.name.trim().to_string();
        if name.chars().count() < 2 || name.chars().count() > 60 {
            continue;
        }
        if is_temporal_or_filler(&name) {
            continue;
        }
        let key = normalize(&name);
        if key.is_empty() || !seen.insert(key) {
            continue;
        }
        out.push(ExtractedEntity {
            name,
            etype: e.etype.trim().to_lowercase(),
        });
        if out.len() >= 8 {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{is_temporal_or_filler, normalize};

    #[test]
    fn dates_and_filler_are_rejected() {
        for s in [
            "January", "this week", "end of year", "Friday", "2026", "12/5", "9:30",
            "January 2026", "deadline", "Project Deadline Approach", "goal",
        ] {
            assert!(is_temporal_or_filler(s), "should reject: {s}");
        }
    }

    #[test]
    fn real_entities_are_kept() {
        for s in [
            "Kepler", "Xarch Labs", "CoreSum", "Gaussian Process", "Stanford University",
            "Yamaha", "SEVIS", "Marchetti", "Mayo Clinic",
        ] {
            assert!(!is_temporal_or_filler(s), "should keep: {s}");
        }
    }

    #[test]
    fn normalize_merges_aliases() {
        assert_eq!(normalize("Stanford University"), normalize("stanford"));
        assert_eq!(normalize("Xarch Labs, Inc."), normalize("xarch labs"));
    }
}
