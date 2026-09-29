//! Edge <-> cloud sync against a REAL Qdrant Server (localhost:6333), with two
//! simulated devices (separate engines, separate device ids). Every test uses
//! its own throwaway collection, deleted afterwards; the app's real collection
//! is never touched. Tests skip (pass with a notice) if no server is reachable.

use std::path::PathBuf;

use mint_core::record::{Memory, MemoryKind, MemorySource, NewMemory, Sensitivity};
use mint_core::sync::{SyncClient, SyncConfig};
use mint_core::MemoryEngine;

struct Device {
    eng: Option<MemoryEngine>,
    dir: PathBuf,
}

impl Device {
    fn new(name: &str) -> Self {
        let models = std::env::var("MINT_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.mint-data/models")
        });
        let dir = std::env::temp_dir().join(format!("mint-sync-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let eng = MemoryEngine::open_with_models(&dir, &models).unwrap();
        Self { eng: Some(eng), dir }
    }
    fn e(&self) -> &MemoryEngine {
        self.eng.as_ref().unwrap()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.eng.take();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

struct Cloud {
    client: SyncClient,
}

impl Cloud {
    /// A throwaway collection, or None when no server is running.
    fn new(name: &str) -> Option<Self> {
        let client = SyncClient::new(SyncConfig {
            url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into()),
            api_key: None,
            collection: format!("mint_test_{name}_{}", std::process::id()),
        });
        if !client.reachable() {
            eprintln!("skipping: no Qdrant Server reachable");
            return None;
        }
        let _ = client.delete_collections();
        Some(Self { client })
    }
    fn payload_texts(&self) -> Vec<String> {
        self.client
            .scroll_all()
            .unwrap()
            .into_iter()
            .map(|p| p["text"].as_str().unwrap_or("").to_string())
            .collect()
    }
}

impl Drop for Cloud {
    fn drop(&mut self) {
        let _ = self.client.delete_collections();
    }
}

fn note(e: &MemoryEngine, text: &str) -> Memory {
    e.add(NewMemory {
        kind: MemoryKind::Note,
        title: text.chars().take(40).collect(),
        text: text.to_string(),
        site_id: String::new(),
        asset_id: String::new(),
        geo: None,
        tags: Vec::new(),
        source: MemorySource::Chat,
        sensitivity: Sensitivity::Shareable,
        parent_id: None,
        topic_id: None,
        due_at: None,
    })
    .unwrap()
}

/// Edit a memory's text as a user would (new updated_at).
fn edit(e: &MemoryEngine, id: &str, text: &str) {
    std::thread::sleep(std::time::Duration::from_millis(5));
    let mut m = e.get(id).unwrap().unwrap();
    m.text = text.to_string();
    m.updated_at = chrono::Utc::now().to_rfc3339();
    e.upsert(&m).unwrap();
}

#[test]
fn sensitive_memories_never_leave_the_device() {
    let Some(cloud) = Cloud::new("policy") else { return };
    let a = Device::new("policy-a");
    note(a.e(), "My wifi password is hunter2");
    note(a.e(), "I'm allergic to peanuts");
    let ok = note(a.e(), "CoreSum TCS is stuck at 0.585");
    let r = a.e().sync(&cloud.client, false).unwrap();
    let texts = cloud.payload_texts();
    assert!(texts.iter().all(|t| !t.contains("hunter2") && !t.contains("peanuts")));
    assert!(texts.iter().any(|t| t == &ok.text));
    assert_eq!(r.withheld, 2, "{r:?}");
    assert_eq!(r.withheld_by.get("secret"), Some(&1));
    assert_eq!(r.withheld_by.get("health"), Some(&1));
}

#[test]
fn making_a_memory_private_retracts_the_cloud_copy() {
    let Some(cloud) = Cloud::new("retract") else { return };
    let a = Device::new("retract-a");
    let m = note(a.e(), "Draft thoughts about the CoreSum grid budget");
    a.e().sync(&cloud.client, false).unwrap();
    assert!(cloud.payload_texts().contains(&m.text));
    let d = a.e().set_sync_override(&m.id, false).unwrap();
    assert!(!d.share);
    let r = a.e().sync(&cloud.client, false).unwrap();
    assert_eq!(r.retracted, 1);
    assert!(!cloud.payload_texts().contains(&m.text));
    // Still fully usable on the device.
    assert!(a.e().get(&m.id).unwrap().is_some());
}

#[test]
fn devices_share_consolidated_knowledge_and_search_the_rest_in_the_cloud() {
    let Some(cloud) = Cloud::new("tiered") else { return };
    let a = Device::new("tiered-a");
    let b = Device::new("tiered-b");
    let (doc, _) = a
        .e()
        .ingest_text(
            "Piano Plan.txt",
            "# Piano practice plan\n\nPractice scales daily and learn Clair de Lune by Debussy.",
            "",
        )
        .unwrap();
    let raw = note(a.e(), "Practiced the Debussy arpeggios for forty minutes tonight");
    a.e().route_and_assign(&raw.id, &raw.text, "");
    a.e().sync(&cloud.client, false).unwrap();

    let r = b.e().sync(&cloud.client, false).unwrap();
    // Consolidated knowledge comes down: the document, its chunks, the topic.
    assert!(b.e().get(&doc.id).unwrap().is_some());
    assert!(!b.e().list_topics().unwrap().is_empty());
    // A's raw note stays in the cloud (tiered pull) ...
    assert!(b.e().get(&raw.id).unwrap().is_none());
    assert!(r.cloud_only >= 1, "{r:?}");
    // ... but B can still reach it through cloud search when online.
    let hits = b.e().cloud_search(&cloud.client, "how long did I practice arpeggios", 3).unwrap();
    assert!(hits.iter().any(|h| h.memory.id == raw.id));
    // Pulled chunks are wired back into the graph.
    let g = b.e().graph_data().unwrap();
    assert!(g.edges.iter().any(|e| e.relation == "part_of" && e.to == doc.id));
}

#[test]
fn one_sided_edits_are_not_conflicts() {
    let Some(cloud) = Cloud::new("onesided") else { return };
    let a = Device::new("onesided-a");
    let b = Device::new("onesided-b");
    let m = note(a.e(), "The DSO meeting is on Tuesday");
    a.e().sync(&cloud.client, false).unwrap();
    b.e().sync(&cloud.client, true).unwrap();
    edit(a.e(), &m.id, "The DSO meeting is on Tuesday at 3pm in room 204");
    a.e().sync(&cloud.client, false).unwrap();
    let r = b.e().sync(&cloud.client, true).unwrap();
    assert_eq!(r.conflicts, 0, "{r:?}");
    assert_eq!(b.e().get(&m.id).unwrap().unwrap().text, "The DSO meeting is on Tuesday at 3pm in room 204");
}

#[test]
fn concurrent_edits_become_a_version_chain_not_data_loss() {
    let Some(cloud) = Cloud::new("conflict") else { return };
    let a = Device::new("conflict-a");
    let b = Device::new("conflict-b");
    let m = note(a.e(), "Project review is on Friday");
    a.e().sync(&cloud.client, false).unwrap();
    b.e().sync(&cloud.client, true).unwrap();

    // Both devices edit the same memory while apart.
    edit(a.e(), &m.id, "Project review moved to Monday");
    edit(b.e(), &m.id, "Project review moved to Wednesday");
    a.e().sync(&cloud.client, false).unwrap();
    let r = b.e().sync(&cloud.client, true).unwrap();
    assert_eq!(r.conflicts, 1, "{r:?}");

    // The newer edit (B's) wins; A's edit survives as an older version.
    let winner = b.e().get(&m.id).unwrap().unwrap();
    assert_eq!(winner.text, "Project review moved to Wednesday");
    let prev = winner.supersedes.clone().expect("winner records the preserved version");
    let kept = b.e().get(&prev).unwrap().unwrap();
    assert_eq!(kept.text, "Project review moved to Monday");
    assert_eq!(kept.superseded_by.as_deref(), Some(m.id.as_str()));

    // A converges on the next sync, with the history intact.
    a.e().sync(&cloud.client, true).unwrap();
    assert_eq!(a.e().get(&m.id).unwrap().unwrap().text, "Project review moved to Wednesday");
    assert!(a.e().get(&prev).unwrap().is_some());
}

#[test]
fn deletions_propagate_between_devices() {
    let Some(cloud) = Cloud::new("delete") else { return };
    let a = Device::new("delete-a");
    let b = Device::new("delete-b");
    let m = note(a.e(), "Temporary note about the lab schedule");
    a.e().sync(&cloud.client, false).unwrap();
    b.e().sync(&cloud.client, true).unwrap();
    assert!(b.e().get(&m.id).unwrap().is_some());
    a.e().delete(&m.id).unwrap();
    a.e().sync(&cloud.client, false).unwrap();
    b.e().sync(&cloud.client, true).unwrap();
    assert!(b.e().get(&m.id).unwrap().is_none());
}

#[test]
fn offline_device_keeps_working() {
    let a = Device::new("offline-a");
    note(a.e(), "Practicing piano scales every morning");
    let dead = SyncClient::new(SyncConfig {
        url: "http://127.0.0.1:9".into(),
        api_key: None,
        collection: "unused".into(),
    });
    assert!(!dead.reachable());
    let t = std::time::Instant::now();
    assert!(a.e().sync(&dead, false).is_err());
    assert!(t.elapsed().as_secs() < 20, "sync must fail fast when the server is down");
    let r = a.e().retrieve("when do I practice piano?", 3).unwrap();
    assert!(!r.is_empty(), "retrieval works fully offline");
}
