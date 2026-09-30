// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
//! mint-core: the local-first cognitive memory engine.
//!
//! Embeds Qdrant Edge (vector store) + fastembed (embeddings), fully offline.
//! Hosted by the Tauri app today; by a headless daemon and other clients later.

pub mod chat;
pub mod conversations;
pub mod documents;
pub mod embed;
pub mod engine;
pub mod entities;
pub mod graph;
pub mod meta;
pub mod ollama;
pub mod policy;
pub mod record;
pub mod sync;

pub use conversations::ConversationStore;
pub use engine::MemoryEngine;
pub use ollama::Ollama;
