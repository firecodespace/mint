//! End-to-end tests for the topic (schema) layer against a real engine:
//! real embeddings, a real Qdrant Edge shard, no LLM. Uses the shared model
//! cache (MINT_MODELS_DIR or <repo>/.mint-data/models).

use std::path::PathBuf;

use mint_core::record::{MemoryKind, MemorySource, NewMemory, Sensitivity};
use mint_core::MemoryEngine;

fn engine(name: &str) -> (MemoryEngine, PathBuf) {
    let models = std::env::var("MINT_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.mint-data/models"));
    let dir = std::env::temp_dir().join(format!("mint-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (MemoryEngine::open_with_models(&dir, &models).unwrap(), dir)
}

fn note(eng: &MemoryEngine, text: &str) -> String {
    eng.add(NewMemory {
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
    .id
}

const PIANO_DOC: &str = "# Piano practice plan\n\nPractice major and minor scales daily. \
Learn Clair de Lune by Debussy. Use a metronome to keep steady tempo while practicing \
arpeggios and chord inversions on the piano.\n\nTake weekly lessons with a piano teacher.";
const TAX_DOC: &str = "# Tax filing checklist\n\nCollect income statements, receipts for \
deductions, and last year's tax return. File the return before the deadline and keep \
copies of every invoice for audit purposes.";

#[test]
fn documents_route_and_chunks_inherit_topic() {
    let (eng, dir) = engine("route");
    let (piano, _) = eng.ingest_text("Piano Plan.txt", PIANO_DOC, "").unwrap();
    let (tax, _) = eng.ingest_text("Taxes.txt", TAX_DOC, "").unwrap();
    let p = eng.get(&piano.id).unwrap().unwrap();
    let t = eng.get(&tax.id).unwrap().unwrap();
    assert!(p.topic_id.is_some() && t.topic_id.is_some());
    assert_ne!(p.topic_id, t.topic_id, "unrelated documents must not share a topic");
    // Chunks carry their document's topic.
    let g = eng.graph_data().unwrap();
    for n in g.nodes.iter().filter(|n| n.parent_id.as_deref() == Some(&piano.id)) {
        assert_eq!(n.topic_id, p.topic_id);
    }
    // A related note joins the piano topic (kNN routing).
    let n = note(&eng, "Practiced piano scales and arpeggios with a metronome today");
    let tid = eng.route_and_assign(&n, "Practiced piano scales and arpeggios with a metronome today", "");
    assert_eq!(tid, p.topic_id);
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn profile_documents_get_their_own_topic() {
    let (eng, dir) = engine("profile");
    let (resume, _) = eng
        .ingest_text("Resume (Aug 2026).pdf", "Skills: Rust, Python. Experience: founder.", "")
        .unwrap();
    let r = eng.get(&resume.id).unwrap().unwrap();
    let topics = eng.list_topics().unwrap();
    let t = topics.iter().find(|t| Some(&t.id) == r.topic_id.as_ref()).unwrap();
    assert_eq!(t.name, "Profile");
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rename_move_merge_and_prune() {
    let (eng, dir) = engine("manage");
    let (piano, _) = eng.ingest_text("Piano Plan.txt", PIANO_DOC, "").unwrap();
    let (tax, _) = eng.ingest_text("Taxes.txt", TAX_DOC, "").unwrap();
    let piano_topic = eng.get(&piano.id).unwrap().unwrap().topic_id.unwrap();
    let tax_topic = eng.get(&tax.id).unwrap().unwrap().topic_id.unwrap();

    // Rename: sticks, and marks the topic as user-named.
    eng.rename_topic(&piano_topic, "Music").unwrap();
    let info = eng.list_topics().unwrap().into_iter().find(|t| t.id == piano_topic).unwrap();
    assert_eq!(info.name, "Music");
    assert!(info.user_named);
    assert!(eng.rename_topic(&piano_topic, "   ").is_err());

    // Move a document: its chunks follow it.
    eng.move_to_topic(&tax.id, &piano_topic).unwrap();
    let g = eng.graph_data().unwrap();
    assert!(g
        .nodes
        .iter()
        .filter(|n| n.id == tax.id || n.parent_id.as_deref() == Some(&tax.id))
        .all(|n| n.topic_id.as_deref() == Some(piano_topic.as_str())));
    // The emptied topic is pruned (not listed).
    eng.prune_empty_topics().unwrap();
    assert!(eng.list_topics().unwrap().iter().all(|t| t.id != tax_topic));

    // Merge: every member moves, the source topic disappears.
    let n = note(&eng, "Quarterly budget review and savings goal");
    let other = eng.route_and_assign(&n, "Quarterly budget review and savings goal", "").unwrap();
    if other != piano_topic {
        eng.merge_topics(&other, &piano_topic).unwrap();
        assert!(eng.get(&other).unwrap().is_none(), "merged topic must be gone");
        assert_eq!(eng.get(&n).unwrap().unwrap().topic_id.as_deref(), Some(piano_topic.as_str()));
        // Graph edges now point at the surviving topic.
        let g = eng.graph_data().unwrap();
        assert!(g.edges.iter().all(|e| e.to != other && e.from != other));
    }
    assert!(eng.merge_topics(&piano_topic, &piano_topic).is_err());
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn organize_files_existing_memories() {
    let (eng, dir) = engine("organize");
    // Memories created without routing (as in a store from before topics).
    // (Very differently worded short notes can embed below the join threshold
    // without entities/LLM; the benchmark tracks that case. Here the notes are
    // clearly about the same practice.)
    let a = note(&eng, "Practice piano scales and arpeggios every morning");
    let b = note(&eng, "Practiced piano scales and arpeggios this morning for twenty minutes");
    assert!(eng.get(&a).unwrap().unwrap().topic_id.is_none());
    let report = eng.organize("").unwrap();
    assert_eq!(report.routed, 2);
    let ta = eng.get(&a).unwrap().unwrap().topic_id;
    let tb = eng.get(&b).unwrap().unwrap().topic_id;
    assert!(ta.is_some());
    assert_eq!(ta, tb, "two piano notes should end up in one topic");
    // Idempotent: nothing left to route.
    assert_eq!(eng.organize("").unwrap().routed, 0);
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn background_entity_enrichment_links_unlinked_memories_once() {
    use mint_core::entities::ExtractedEntity;
    let (eng, dir) = engine("enrich");
    let a = note(&eng, "Visited the Kepler exhibit at the science museum");
    let (doc, _) = eng.ingest_text("Piano Plan.txt", PIANO_DOC, "").unwrap();
    // Nothing has entities yet (no LLM in this test): both are in the backlog.
    let backlog = eng.entity_backlog(10).unwrap();
    let ids: Vec<&str> = backlog.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ids.contains(&a.as_str()) && ids.contains(&doc.id.as_str()));
    // The document is represented by its content, not just the title.
    let doc_text = &backlog.iter().find(|(id, _)| id == &doc.id).unwrap().1;
    assert!(doc_text.contains("Debussy"));
    // Link extracted entities (as the background job does after the LLM call).
    let ents = vec![ExtractedEntity { name: "Kepler".into(), etype: "concept".into() }];
    let linked = eng.link_extracted_entities(&a, &ents).unwrap();
    assert_eq!(linked.len(), 1);
    // A memory with no entities found is still marked done: no repeat work.
    eng.link_extracted_entities(&doc.id, &[]).unwrap();
    assert!(eng.entity_backlog(10).unwrap().is_empty());
    let g = eng.graph_data().unwrap();
    assert!(g.edges.iter().any(|e| e.relation == "mentions" && e.from == a));
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn retrieval_reports_the_active_subject() {
    let (eng, dir) = engine("retrieve");
    let (piano, _) = eng.ingest_text("Piano Plan.txt", PIANO_DOC, "").unwrap();
    eng.ingest_text("Taxes.txt", TAX_DOC, "").unwrap();
    let piano_topic = eng.get(&piano.id).unwrap().unwrap().topic_id;
    let ctx = eng.retrieve_context("which Debussy piece am I learning?", 4).unwrap();
    assert!(!ctx.results.is_empty());
    // Top hit is the piano document itself or one of its chunks.
    let top = &ctx.results[0].memory;
    assert!(top.id == piano.id || top.parent_id.as_deref() == Some(piano.id.as_str()));
    assert_eq!(ctx.topics.first().map(|t| Some(t.id.clone())), Some(piano_topic));
    // Scores are normalized for display.
    assert!((ctx.results[0].score - 1.0).abs() < 1e-5);
    assert!(ctx.results.iter().all(|r| r.score <= 1.0 + 1e-5));
    drop(eng);
    let _ = std::fs::remove_dir_all(dir);
}
