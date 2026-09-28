//! mint-core: the local-first cognitive memory engine.
//!
//! Embeds Qdrant Edge (vector store) + fastembed (embeddings), fully offline.
//! Hosted by the Tauri app today; by a headless daemon and other clients later.

pub mod chat;
pub mod conversations;
pub mod embed;
pub mod engine;
pub mod ollama;
pub mod record;

pub use conversations::ConversationStore;
pub use engine::MemoryEngine;
pub use ollama::Ollama;
