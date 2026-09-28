//! Local Ollama client. Localhost only, fully offline. Uses `ureq` (already in
//! our dependency tree via fastembed) so we add no heavy new dependencies.
//!
//! Chat uses Ollama's `think: true`, so reasoning-capable models (e.g. qwen3)
//! return their chain-of-thought in a separate `thinking` field — which we route
//! to the UI's Cognition panel, separate from the visible answer.

use std::io::{BufRead, BufReader, Read};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Strong reasoning model with visible thinking — default for chat.
pub const DEFAULT_CHAT_MODEL: &str = "qwen3:8b";
/// Small, fast model — default for background extraction/consolidation.
pub const DEFAULT_FAST_MODEL: &str = "llama3.2:latest";

fn base_url() -> String {
    match std::env::var("OLLAMA_HOST") {
        Ok(h) if !h.trim().is_empty() => {
            if h.starts_with("http") {
                h
            } else {
                format!("http://{h}")
            }
        }
        _ => "http://localhost:11434".to_string(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

impl ChatMessage {
    pub fn system(c: impl Into<String>) -> Self {
        Self { role: "system".into(), content: c.into() }
    }
    pub fn user(c: impl Into<String>) -> Self {
        Self { role: "user".into(), content: c.into() }
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: c.into() }
    }
}

/// A streamed delta from the model during a chat turn.
pub enum Delta<'a> {
    /// The model's private reasoning (from `thinking`) -> Cognition panel.
    Thinking(&'a str),
    /// The visible answer text -> chat bubble.
    Answer(&'a str),
}

pub struct Ollama {
    base: String,
}

impl Default for Ollama {
    fn default() -> Self {
        Self::new()
    }
}

impl Ollama {
    pub fn new() -> Self {
        Self { base: base_url() }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// Is the local Ollama server reachable?
    pub fn is_up(&self) -> bool {
        ureq::get(format!("{}/api/tags", self.base)).call().is_ok()
    }

    /// Names of locally-pulled models.
    pub fn list_models(&self) -> Result<Vec<String>> {
        let mut resp = ureq::get(format!("{}/api/tags", self.base))
            .call()
            .map_err(|e| anyhow!("ollama not reachable: {e}"))?;
        let mut body = String::new();
        resp.body_mut().as_reader().read_to_string(&mut body)?;
        let v: serde_json::Value = serde_json::from_str(&body)?;
        let models = v["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    /// Stream a chat completion. `on_delta` is called for each token chunk,
    /// tagged as thinking or answer. Returns the full visible answer.
    pub fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        think: bool,
        mut on_delta: impl FnMut(Delta),
    ) -> Result<String> {
        let req = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "think": think,
        });
        let body = serde_json::to_string(&req)?;
        let resp = ureq::post(format!("{}/api/chat", self.base))
            .header("Content-Type", "application/json")
            .send(body)
            .map_err(|e| anyhow!("ollama chat failed: {e}"))?;

        let reader = BufReader::new(resp.into_body().into_reader());
        let mut answer = String::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(err) = v["error"].as_str() {
                return Err(anyhow!("ollama error: {err}"));
            }
            if let Some(t) = v["message"]["thinking"].as_str() {
                if !t.is_empty() {
                    on_delta(Delta::Thinking(t));
                }
            }
            if let Some(c) = v["message"]["content"].as_str() {
                if !c.is_empty() {
                    answer.push_str(c);
                    on_delta(Delta::Answer(c));
                }
            }
            if v["done"].as_bool() == Some(true) {
                break;
            }
        }
        Ok(answer)
    }

    /// Non-streaming generation, used for extraction/consolidation. When
    /// `json` is true, asks Ollama to constrain output to valid JSON.
    pub fn generate(
        &self,
        model: &str,
        system: Option<&str>,
        prompt: &str,
        json: bool,
    ) -> Result<String> {
        let mut messages: Vec<ChatMessage> = Vec::new();
        if let Some(s) = system {
            messages.push(ChatMessage::system(s));
        }
        messages.push(ChatMessage::user(prompt));

        let mut req = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "think": false,
        });
        if json {
            req["format"] = serde_json::Value::String("json".into());
        }
        let body = serde_json::to_string(&req)?;
        let mut resp = ureq::post(format!("{}/api/chat", self.base))
            .header("Content-Type", "application/json")
            .send(body)
            .map_err(|e| anyhow!("ollama generate failed: {e}"))?;
        let mut out = String::new();
        resp.body_mut().as_reader().read_to_string(&mut out)?;
        let v: serde_json::Value = serde_json::from_str(&out)?;
        Ok(v["message"]["content"].as_str().unwrap_or_default().to_string())
    }
}
