use mint_core::record::*;
use mint_core::MemoryEngine;
use std::path::PathBuf;
fn main() {
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.mint-data/models");
    let dir = std::env::temp_dir().join(format!("mint-diag-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let eng = MemoryEngine::open_with_models(&dir, &models).unwrap();
    let mk = |t: &str| eng.add(NewMemory { kind: MemoryKind::Note, title: t.into(), text: t.into(), site_id: String::new(), asset_id: String::new(), geo: None, tags: vec![], source: MemorySource::Chat, sensitivity: Sensitivity::Shareable, parent_id: None, topic_id: None, due_at: None }).unwrap();
    let a = mk("Learn Clair de Lune on the piano by summer");
    let _b = mk("Practice piano scales every morning");
    let r = eng.search(SearchRequest { query: a.text.clone(), mode: SearchMode::Dense, limit: 3, site_id: None, kind: None }).unwrap();
    for x in r.results { println!("note-sim {:.3} {}", x.score, x.memory.title); }
    let piano = "# Piano practice plan\n\nPractice major and minor scales daily. Learn Clair de Lune by Debussy. Use a metronome to keep steady tempo while practicing arpeggios and chord inversions on the piano.\n\nTake weekly lessons with a piano teacher.";
    let (doc, n) = eng.ingest_text("Piano Plan.txt", piano, "").unwrap();
    let d = eng.get(&doc.id).unwrap().unwrap();
    println!("doc topic {:?}, chunks {}", d.topic_id, n);
    let g = eng.graph_data().unwrap();
    for x in g.nodes.iter().filter(|x| x.kind != "note") { println!("node {} kind={} topic={:?}", x.label, x.kind, x.topic_id); }
    let ctx = eng.retrieve_context("which Debussy piece am I learning?", 4).unwrap();
    for x in &ctx.results { println!("hit {:.3} {:?} topic={:?}", x.score, x.memory.kind, x.memory.topic_id); }
    println!("topics {:?}", ctx.topics);
    drop(eng); let _ = std::fs::remove_dir_all(dir);
}
