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
