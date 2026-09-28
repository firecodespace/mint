export function HowItWorks() {
  return (
    <div className="how">
      <div className="how-inner">
        <h1>How Mint works</h1>
        <p className="how-lead">
          Mint is an offline-first AI memory and intelligence platform. Everything —
          embeddings, vector search, and language reasoning — runs on this device. No
          network is required after the first launch, and nothing you say leaves the
          machine.
        </p>

        <details open>
          <summary>The problem</summary>
          <div className="how-body">
            <p>
              AI at the edge must search and reason over locally generated information
              without depending on the cloud: latency matters, connectivity is
              intermittent, and sensitive data cannot always leave the device. The
              challenge is to build an offline-first application that maintains local
              semantic memory, retrieves it with low latency, keeps working offline, and
              synchronizes intelligently when a connection returns.
            </p>
          </div>
        </details>

        <details open>
          <summary>The approach: one event bus, one cognitive core</summary>
          <div className="how-body">
            <p>
              Every input — a chat turn, a dropped document, a git commit, a captured
              note — is the same primitive: a memory event flowing through a local
              pipeline. One cognitive core processes them all; everything else is an
              ingestion adapter or a display surface.
            </p>
            <pre className="how-diagram">{`chat / docs / git / notes
      |
      v
  normalize  ->  extract (LLM)  ->  embed (dense + sparse)
      ->  route / dedup  ->  store & update (Qdrant Edge)
      ->  consolidate (LLM)  ->  retrieve (hybrid)  ->  answer`}</pre>
          </div>
        </details>

        <details>
          <summary>The stack (all local)</summary>
          <div className="how-body">
            <ul className="how-list">
              <li>
                <strong>Qdrant Edge</strong> — an embedded, in-process vector database.
                No server, no Docker. It stores every memory as dense + sparse vectors on
                disk and runs hybrid search.
              </li>
              <li>
                <strong>fastembed</strong> — local ONNX embeddings (all-MiniLM-L6-v2,
                384d). Downloaded once, then fully offline.
              </li>
              <li>
                <strong>Ollama</strong> — the local language model. It answers, and it
                distills durable memories from what you say. Reasoning-capable models
                stream their thinking into the Cognition panel.
              </li>
              <li>
                <strong>Tauri + React</strong> — one native app hosting the engine and the
                UI. A localhost API (in progress) lets a CLI and IDE plugin use the same
                core.
              </li>
            </ul>
          </div>
        </details>

        <details>
          <summary>How retrieval works</summary>
          <div className="how-body">
            <p>
              A query is embedded two ways: a dense vector (semantic meaning) and a sparse
              vector (BM25 keywords). Qdrant Edge runs both searches and fuses the results
              with Reciprocal Rank Fusion, so you get meaning-based and exact-word matches
              together — typically in a few milliseconds, with no network.
            </p>
          </div>
        </details>

        <details>
          <summary>What becomes a memory (and what does not)</summary>
          <div className="how-body">
            <p>
              After each turn a fast model extracts only <em>new, durable facts you
              stated</em> — your name, decisions, preferences, plans, deadlines. Questions,
              greetings, and restatements are ignored. Before storing, a similarity check
              skips near-duplicates, so asking about something you already told Mint does
              not create a second memory.
            </p>
          </div>
        </details>

        <details>
          <summary>What is delivered, and what is next</summary>
          <div className="how-body">
            <ul className="how-list">
              <li><strong>Delivered:</strong> offline hybrid retrieval engine, persistent on-device memory, local chat with streamed thinking, automatic memory capture with dedup, conversation history, and a memory graph.</li>
              <li><strong>Next — intelligence:</strong> layered memory (episodic/semantic), consolidation and decay, multimodal ingest (documents and images), proactive tasks and dates.</li>
              <li><strong>Next — developer platform:</strong> localhost API, a CLI with git hooks, inline capture, and a VS Code plugin.</li>
              <li><strong>Next — sync:</strong> edge-to-cloud synchronization to a Qdrant Server with conflict resolution and a sync-status view, for backup and cross-device memory.</li>
            </ul>
          </div>
        </details>
      </div>
    </div>
  );
}
