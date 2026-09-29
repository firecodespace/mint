//! Entity extraction: pull the key named entities from a piece of text so they
//! can become connective hub nodes in the knowledge graph. Local (Ollama).

use anyhow::Result;
use serde::Deserialize;

use crate::ollama::Ollama;

const SYSTEM: &str = "Extract the key named entities from the text for a knowledge graph: \
specific people, organizations, places, projects, products, and important domain concepts. \
Return ONLY JSON of the form {\"entities\":[\"Canonical Name\", ...]}. Rules: use short \
canonical names; at most 8; skip generic words, pronouns, dates, and filler. If there are no \
meaningful entities, return {\"entities\":[]}.";

#[derive(Debug, Deserialize)]
struct EntityResult {
    #[serde(default)]
    entities: Vec<String>,
}

/// Extract up to 8 canonical entity names from `text`.
pub fn extract(ollama: &Ollama, model: &str, text: &str) -> Result<Vec<String>> {
    let sample: String = text.chars().take(3000).collect();
    let raw = ollama.generate(model, Some(SYSTEM), &sample, true)?;
    let parsed: EntityResult = serde_json::from_str(&raw).unwrap_or(EntityResult {
        entities: Vec::new(),
    });

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for name in parsed.entities {
        let name = name.trim().to_string();
        if name.chars().count() < 2 || name.chars().count() > 60 {
            continue;
        }
        let key = name.to_lowercase();
        if seen.insert(key) {
            out.push(name);
        }
        if out.len() >= 8 {
            break;
        }
    }
    Ok(out)
}
