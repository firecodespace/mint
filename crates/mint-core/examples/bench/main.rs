//! Mint retrieval + memory-organization benchmark.
//!
//! Builds a throwaway engine, ingests a labeled corpus, and measures:
//!   - retrieval quality (Recall@k, MRR, nDCG@k, Hit@1, Precision@k) for the raw
//!     hybrid search and for the chat retrieval pipeline,
//!   - latency (p50 / p95) and ingestion throughput,
//!   - topic-routing quality (pairwise precision / recall / F1 vs. ground truth),
//!   - capture-guard accuracy (which chat messages become memories).
//!
//! Deterministic and offline by default (no LLM). Pass `--llm <model>` to also
//! exercise the Ollama-backed steps (entities, topic naming, topic summaries).
//!
//!   cargo run --release --example bench -- --label baseline
//!   cargo run --release --example bench -- --label v2 --compare baseline

mod corpus;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use mint_core::record::{
    MemoryKind, MemorySource, NewMemory, SearchMode, SearchRequest, SearchResult, Sensitivity,
};
use mint_core::MemoryEngine;
use serde_json::{json, Value};

const K: usize = 6;
const LATENCY_REPS: usize = 5;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let label = arg(&args, "--label").unwrap_or_else(|| "run".into());
    let compare = arg(&args, "--compare");
    let llm = arg(&args, "--llm").unwrap_or_default();
    let scale: usize = arg(&args, "--scale").and_then(|s| s.parse().ok()).unwrap_or(0);
    let models = arg(&args, "--models")
        .or_else(|| std::env::var("MINT_MODELS_DIR").ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.mint-data/models")
        });

    let dir = std::env::temp_dir().join(format!("mint-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let report = {
        let mut eng = MemoryEngine::open_with_models(&dir, &models)?;
        // --tune topic_join=0.45,offtopic_penalty=0.5 (see engine::Tuning)
        if let Some(spec) = arg(&args, "--tune") {
            let mut t = eng.tuning();
            for kv in spec.split(',') {
                let Some((k, v)) = kv.split_once('=') else { continue };
                let v: f32 = v.trim().parse()?;
                match k.trim() {
                    "topic_join" => t.topic_join = v,
                    "topic_join_direct" => t.topic_join_direct = v,
                    "activation_min" => t.activation_min = v,
                    "focus_conf" => t.focus_conf = v,
                    "offtopic_penalty" => t.offtopic_penalty = v,
                    "topic_weight" => t.topic_weight = v,
                    "named_weight" => t.named_weight = v,
                    "profile_weight" => t.profile_weight = v,
                    "rrf_k" => t.rrf_k = v,
                    "per_doc_cap" => t.per_doc_cap = v as usize,
                    other => anyhow::bail!("unknown tuning key: {other}"),
                }
            }
            eng.set_tuning(t);
        }
        let tuning = format!("{:?}", eng.tuning());
        let mut r = run(&eng, &label, &llm, scale)?;
        r["tuning"] = json!(tuning);
        r
    };
    let _ = std::fs::remove_dir_all(&dir);

    print_report(&report);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench-results");
    std::fs::create_dir_all(&out_dir)?;
    let path = out_dir.join(format!("{label}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("\nsaved {}", path.display());

    if let Some(prev) = compare {
        let prev_path = out_dir.join(format!("{prev}.json"));
        match std::fs::read(&prev_path) {
            Ok(bytes) => print_compare(&serde_json::from_slice(&bytes)?, &report),
            Err(_) => println!("(no results at {} to compare)", prev_path.display()),
        }
    }
    Ok(())
}

fn new_mem(kind: MemoryKind, title: &str, text: &str, parent: Option<String>) -> NewMemory {
    NewMemory {
        kind,
        title: title.to_string(),
        text: text.to_string(),
        site_id: String::new(),
        asset_id: String::new(),
        geo: None,
        tags: Vec::new(),
        source: MemorySource::Chat,
        sensitivity: Sensitivity::Shareable,
        parent_id: parent,
        topic_id: None,
        due_at: None,
    }
}

fn run(eng: &MemoryEngine, label: &str, llm: &str, scale: usize) -> Result<Value> {
    // ---- scale: distractor documents (incl. hard negatives) ---------------
    // Ingested first so the labeled corpus must be found among them. Never
    // spends LLM calls (entity extraction on hundreds of docs is not the point).
    // id -> ground-truth item key (documents map their chunks via parent_id),
    // and item key -> true subject. Distractors are items too, so routing
    // quality counts a real paper being absorbed into a distractor topic.
    let mut item_of: HashMap<String, String> = HashMap::new();
    let mut true_topic: HashMap<String, String> = HashMap::new();

    let ts = Instant::now();
    let mut scale_chunks = 0usize;
    for (i, (title, text, domain)) in corpus::filler_docs(scale).into_iter().enumerate() {
        let (doc, n) = eng.ingest_text(&title, &text, "")?;
        scale_chunks += n;
        let key = format!("filler-{i}");
        item_of.insert(doc.id.clone(), key.clone());
        true_topic.insert(key, format!("filler:{domain}"));
    }
    let scale_ms = ts.elapsed().as_secs_f64() * 1000.0;

    // ---- ingest -----------------------------------------------------------
    let mut chunks = 0usize;

    let t0 = Instant::now();
    for d in corpus::DOCS {
        let (doc, n) = eng.ingest_text(d.title, d.text, llm)?;
        chunks += n;
        item_of.insert(doc.id.clone(), d.key.to_string());
        true_topic.insert(d.key.to_string(), d.topic.to_string());
    }
    let docs_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Phase 3 (LLM mode): distill rolling topic summaries from the documents
    // BEFORE notes arrive, so notes route against what each subject is about.
    let mut summaries_ms = 0.0;
    let mut summarized = 0usize;
    if !llm.is_empty() {
        let t = Instant::now();
        summarized = eng.refresh_due_topics(llm, 1, 50);
        summaries_ms = t.elapsed().as_secs_f64() * 1000.0;
    }

    let t1 = Instant::now();
    for n in corpus::NOTES {
        let title: String = n.text.chars().take(48).collect();
        let m = eng.add(new_mem(MemoryKind::Note, &title, n.text, None))?;
        // Mirror the chat capture path in LLM mode: link entities, then route.
        if !llm.is_empty() {
            let _ = eng.attach_entities(&m.id, &m.text, llm);
        }
        // Mirror the chat capture path: route every captured note to a topic.
        eng.route_and_assign(&m.id, &m.text, llm);
        item_of.insert(m.id.clone(), n.key.to_string());
        true_topic.insert(n.key.to_string(), n.topic.to_string());
    }
    let notes_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Periodic maintenance: re-home stragglers now that every subject exists.
    let tc = Instant::now();
    let rehomed = eng.consolidate_topics()?;
    let consolidate_ms = tc.elapsed().as_secs_f64() * 1000.0;

    // Optional LLM-backed maintenance (topic summaries etc.) when available.
    let mut maintenance_ms = 0.0;
    if !llm.is_empty() {
        let t = Instant::now();
        let _ = eng.run_maintenance(llm);
        maintenance_ms = t.elapsed().as_secs_f64() * 1000.0;
    }

    let all = eng.list()?;
    let by_id: HashMap<String, _> = all.iter().map(|m| (m.id.clone(), m.clone())).collect();
    let group_of = |r: &SearchResult| -> Option<String> {
        if let Some(k) = item_of.get(&r.memory.id) {
            return Some(k.clone());
        }
        r.memory
            .parent_id
            .as_ref()
            .and_then(|p| item_of.get(p))
            .cloned()
    };

    // ---- topic routing quality -------------------------------------------
    // Pairwise agreement between predicted topic_id and ground-truth subject
    // over every top-level item (documents + notes), distractors included.
    let items: Vec<(String, String, Option<String>)> = item_of
        .iter()
        .filter_map(|(id, key)| {
            let tt = true_topic.get(key)?;
            let pred = by_id.get(id).and_then(|m| m.topic_id.clone());
            Some((key.clone(), tt.clone(), pred))
        })
        .collect();
    let is_filler = |k: &str| k.starts_with("filler-");
    let pairwise = |only_labeled: bool| -> (f64, f64, f64) {
        let sel: Vec<&(String, String, Option<String>)> =
            items.iter().filter(|x| !only_labeled || !is_filler(&x.0)).collect();
        let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
        for i in 0..sel.len() {
            for j in (i + 1)..sel.len() {
                let same_true = sel[i].1 == sel[j].1;
                let same_pred = sel[i].2.is_some() && sel[i].2 == sel[j].2;
                match (same_true, same_pred) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fneg += 1,
                    _ => {}
                }
            }
        }
        let p = ratio(tp, tp + fp);
        let r = ratio(tp, tp + fneg);
        (p, r, f1(p, r))
    };
    let (t_prec, t_rec, t_f1) = pairwise(true);
    let (a_prec, a_rec, a_f1) = pairwise(false);
    let pred_topics: HashSet<_> = items.iter().filter_map(|x| x.2.clone()).collect();
    let true_topics: HashSet<_> = items.iter().map(|x| x.1.clone()).collect();
    let unrouted = items.iter().filter(|x| x.2.is_none()).count();
    // A topic is impure when it mixes true subjects; a labeled item is absorbed
    // when it shares a topic with a distractor from another subject.
    let mut impure = 0usize;
    let mut absorbed: Vec<String> = Vec::new();
    let mut topic_listing: Vec<Value> = Vec::new();
    for tid in &pred_topics {
        let name = by_id.get(tid).map(|m| m.title.clone()).unwrap_or_default();
        let members: Vec<&(String, String, Option<String>)> =
            items.iter().filter(|x| x.2.as_ref() == Some(tid)).collect();
        let subjects: HashSet<&String> = members.iter().map(|x| &x.1).collect();
        if subjects.len() > 1 {
            impure += 1;
            for m in &members {
                if !is_filler(&m.0)
                    && members.iter().any(|o| is_filler(&o.0) && o.1 != m.1)
                {
                    absorbed.push(m.0.clone());
                }
            }
        }
        let mut labeled: Vec<String> = members
            .iter()
            .filter(|x| !is_filler(&x.0))
            .map(|x| x.0.clone())
            .collect();
        labeled.sort();
        let fillers = members.len() - labeled.len();
        if !labeled.is_empty() {
            let summary = by_id
                .get(tid)
                .filter(|m| m.text != m.title)
                .map(|m| m.text.clone())
                .unwrap_or_default();
            topic_listing.push(json!({
                "name": name, "members": labeled, "fillers": fillers, "summary": summary,
            }));
        }
    }
    absorbed.sort();

    // Routing diagnostic: for each labeled item NOT filed with the majority of
    // its true subject, show its nearest filed neighbours and best topic-node
    // match (dense cosine) so thresholds are tuned on evidence, not guesses.
    let mut majority: HashMap<String, (Option<String>, usize)> = HashMap::new();
    {
        let mut counts: HashMap<(String, Option<String>), usize> = HashMap::new();
        for x in items.iter().filter(|x| !is_filler(&x.0)) {
            *counts.entry((x.1.clone(), x.2.clone())).or_insert(0) += 1;
        }
        for ((subject, tid), n) in counts {
            let e = majority.entry(subject).or_insert((None, 0));
            if n > e.1 {
                *e = (tid, n);
            }
        }
    }
    let mut diag: Vec<Value> = Vec::new();
    for (id, key) in &item_of {
        if is_filler(key) {
            continue;
        }
        let Some(subject) = true_topic.get(key) else { continue };
        let Some(m) = by_id.get(id) else { continue };
        let Some((maj, n)) = majority.get(subject) else { continue };
        if *n < 2 || &m.topic_id == maj {
            continue;
        }
        let near = eng.search(SearchRequest {
            query: m.text.clone(),
            mode: SearchMode::Dense,
            limit: 6,
            site_id: None,
            kind: None,
        })?;
        let neighbours: Vec<String> = near
            .results
            .iter()
            .filter(|r| r.memory.id != m.id && r.memory.kind != MemoryKind::Topic)
            .take(3)
            .map(|r| {
                let g = item_of
                    .get(&r.memory.id)
                    .or_else(|| r.memory.parent_id.as_ref().and_then(|p| item_of.get(p)))
                    .cloned()
                    .unwrap_or_else(|| "-".into());
                format!("{g} {:.2}", r.score)
            })
            .collect();
        let topic_hit = eng
            .search(SearchRequest {
                query: m.text.clone(),
                mode: SearchMode::Dense,
                limit: 1,
                site_id: None,
                kind: Some(MemoryKind::Topic),
            })?
            .results
            .first()
            .map(|r| format!("{} {:.2}", r.memory.title, r.score))
            .unwrap_or_default();
        diag.push(json!({ "item": key, "neighbours": neighbours, "best_topic": topic_hit }));
    }

    // ---- capture guard -----------------------------------------------------
    let mut cap_ok = 0;
    let mut cap_fail: Vec<Value> = Vec::new();
    for c in corpus::CAPTURE_CASES {
        let captured = !mint_core::chat::is_query_only(c.text);
        if captured == c.capture {
            cap_ok += 1;
        } else {
            cap_fail.push(json!({ "text": c.text, "expected_capture": c.capture }));
        }
    }
    let cap_acc = ratio(cap_ok, corpus::CAPTURE_CASES.len());

    // ---- retrieval quality + latency --------------------------------------
    let mut pipelines = serde_json::Map::new();
    let mut per_query: Vec<Value> = Vec::new();
    for pipe in ["search", "retrieve"] {
        let mut agg = Agg::default();
        let mut lat: Vec<f64> = Vec::new();
        for q in corpus::QUERIES {
            let results = run_pipeline(eng, pipe, q.text)?;
            // latency: warm reps
            for _ in 0..LATENCY_REPS {
                let t = Instant::now();
                let _ = run_pipeline(eng, pipe, q.text)?;
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let groups: Vec<Option<String>> = results.iter().map(|r| group_of(r)).collect();
            let s = score(&groups, q.relevant);
            agg.add(&s);
            per_query.push(json!({
                "pipeline": pipe,
                "id": q.id,
                "query": q.text,
                "recall": s.recall, "mrr": s.mrr, "ndcg": s.ndcg,
                "hit1": s.hit1, "precision": s.precision,
                "top": groups.iter().zip(results.iter()).map(|(g, r)| {
                    format!("{} [{}]", g.clone().unwrap_or_else(|| "-".into()), kind_label(&r.memory.kind))
                }).collect::<Vec<_>>(),
            }));
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pipelines.insert(
            pipe.to_string(),
            json!({
                "recall": agg.mean(|s| s.recall),
                "mrr": agg.mean(|s| s.mrr),
                "ndcg": agg.mean(|s| s.ndcg),
                "hit1": agg.mean(|s| s.hit1),
                "precision": agg.mean(|s| s.precision),
                "p50_ms": pct(&lat, 0.50),
                "p95_ms": pct(&lat, 0.95),
            }),
        );
    }

    // ---- ingestion A/B (after evaluation, so it cannot affect quality) -----
    // Same long document stored (a) the old way, one embed+upsert+flush per
    // chunk, and (b) through ingest_text's batched path.
    let long = corpus::long_doc(60);
    let ab_chunks = mint_core::documents::chunk_text(&long);
    let t_old = Instant::now();
    let parent = eng.add(new_mem(MemoryKind::Document, "ab-old.txt", &long[..400], None))?;
    for (i, c) in ab_chunks.iter().enumerate() {
        eng.add(new_mem(
            MemoryKind::DocChunk,
            &format!("ab-old.txt [{}]", i + 1),
            c,
            Some(parent.id.clone()),
        ))?;
    }
    let old_ms = t_old.elapsed().as_secs_f64() * 1000.0;
    let t_new = Instant::now();
    eng.ingest_text("ab-new.txt", &long, "")?;
    let new_ms = t_new.elapsed().as_secs_f64() * 1000.0;

    Ok(json!({
        "label": label,
        "when": chrono::Utc::now().to_rfc3339(),
        "llm": llm,
        "k": K,
        "scale": {
            "docs": scale, "chunks": scale_chunks, "ms": scale_ms,
            "chunks_per_sec": scale_chunks as f64 / (scale_ms / 1000.0).max(1e-9),
        },
        "ingest_ab": {
            "chunks": ab_chunks.len(), "old_ms": old_ms, "new_ms": new_ms,
            "speedup": old_ms / new_ms.max(1e-9),
        },
        "ingest": {
            "docs": corpus::DOCS.len(), "chunks": chunks, "notes": corpus::NOTES.len(),
            "docs_ms": docs_ms, "notes_ms": notes_ms,
            "chunks_per_sec": chunks as f64 / (docs_ms / 1000.0).max(1e-9),
            "maintenance_ms": maintenance_ms,
            "topic_summaries": summarized, "topic_summaries_ms": summaries_ms,
            "rehomed": rehomed, "consolidate_ms": consolidate_ms,
        },
        "topics": {
            "predicted": pred_topics.len(), "true": true_topics.len(), "unrouted": unrouted,
            "pair_precision": t_prec, "pair_recall": t_rec, "pair_f1": t_f1,
            "all_precision": a_prec, "all_recall": a_rec, "all_f1": a_f1,
            "impure": impure, "absorbed": absorbed,
            "listing": topic_listing, "split_diagnostics": diag,
        },
        "capture": { "accuracy": cap_acc, "failures": cap_fail },
        "pipelines": pipelines,
        "per_query": per_query,
    }))
}

fn run_pipeline(eng: &MemoryEngine, pipe: &str, q: &str) -> Result<Vec<SearchResult>> {
    let mut r = match pipe {
        "search" => {
            eng.search(SearchRequest {
                query: q.to_string(),
                mode: SearchMode::Hybrid,
                limit: K,
                site_id: None,
                kind: None,
            })?
            .results
        }
        _ => eng.retrieve(q, K)?,
    };
    // Mirror the chat path: bare entity names never ground an answer.
    r.retain(|x| x.memory.kind != MemoryKind::Entity);
    r.truncate(K);
    Ok(r)
}

fn kind_label(k: &MemoryKind) -> &'static str {
    match k {
        MemoryKind::DocChunk => "chunk",
        MemoryKind::Document => "doc",
        MemoryKind::Note => "note",
        MemoryKind::Topic => "topic",
        MemoryKind::Summary => "summary",
        MemoryKind::Entity => "entity",
        _ => "other",
    }
}

#[derive(Default, Clone)]
struct Scores {
    recall: f64,
    mrr: f64,
    ndcg: f64,
    hit1: f64,
    precision: f64,
}

#[derive(Default)]
struct Agg(Vec<Scores>);
impl Agg {
    fn add(&mut self, s: &Scores) {
        self.0.push(s.clone());
    }
    fn mean(&self, f: impl Fn(&Scores) -> f64) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }
        self.0.iter().map(f).sum::<f64>() / self.0.len() as f64
    }
}

/// Score a ranked list of result groups against graded judgments. Each relevant
/// item earns gain only on its FIRST appearance, so five chunks of one document
/// do not count as five answers (rewards covering distinct relevant sources).
fn score(groups: &[Option<String>], relevant: &[(&str, u8)]) -> Scores {
    let grade: HashMap<&str, u8> = relevant.iter().copied().collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut dcg = 0.0;
    let mut first_rel: Option<usize> = None;
    let mut rel_hits = 0usize;
    for (i, g) in groups.iter().enumerate() {
        let Some(g) = g else { continue };
        if let Some(&gr) = grade.get(g.as_str()) {
            rel_hits += 1;
            if first_rel.is_none() {
                first_rel = Some(i);
            }
            if seen.insert(g.clone()) {
                dcg += (2f64.powi(gr as i32) - 1.0) / ((i + 2) as f64).log2();
            }
        }
    }
    let mut ideal: Vec<u8> = relevant.iter().map(|x| x.1).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(K)
        .enumerate()
        .map(|(i, &gr)| (2f64.powi(gr as i32) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    let found = seen.len();
    Scores {
        recall: ratio(found, relevant.len()),
        mrr: first_rel.map(|i| 1.0 / (i as f64 + 1.0)).unwrap_or(0.0),
        ndcg: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        hit1: if first_rel == Some(0) { 1.0 } else { 0.0 },
        precision: ratio(rel_hits, groups.len().max(1)),
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

fn f1(p: f64, r: f64) -> f64 {
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn print_report(r: &Value) {
    println!("\n=== Mint benchmark: {} (k={}) ===", r["label"], r["k"]);
    let ing = &r["ingest"];
    println!(
        "ingest: {} docs -> {} chunks in {:.0} ms ({:.1} chunks/s); {} notes in {:.0} ms",
        ing["docs"], ing["chunks"], f(&ing["docs_ms"]), f(&ing["chunks_per_sec"]),
        ing["notes"], f(&ing["notes_ms"])
    );
    println!(
        "consolidation: {} memories re-homed in {:.0} ms",
        ing["rehomed"], f(&ing["consolidate_ms"])
    );
    if f(&ing["topic_summaries"]) > 0.0 {
        println!(
            "topic summaries: {} distilled in {:.0} ms; maintenance {:.0} ms",
            ing["topic_summaries"], f(&ing["topic_summaries_ms"]), f(&ing["maintenance_ms"])
        );
    }
    let sc = &r["scale"];
    if f(&sc["docs"]) > 0.0 {
        println!(
            "scale: {} distractor docs -> {} chunks in {:.0} ms ({:.1} chunks/s)",
            sc["docs"], sc["chunks"], f(&sc["ms"]), f(&sc["chunks_per_sec"])
        );
    }
    let ab = &r["ingest_ab"];
    println!(
        "ingest A/B ({} chunks): per-chunk {:.0} ms vs batched {:.0} ms -> {:.1}x faster",
        ab["chunks"], f(&ab["old_ms"]), f(&ab["new_ms"]), f(&ab["speedup"])
    );
    let t = &r["topics"];
    println!(
        "topics: predicted {} / true {} (unrouted {}), pairwise P {:.3} R {:.3} F1 {:.3}",
        t["predicted"], t["true"], t["unrouted"],
        f(&t["pair_precision"]), f(&t["pair_recall"]), f(&t["pair_f1"])
    );
    println!(
        "        all items incl. distractors: P {:.3} R {:.3} F1 {:.3}; impure topics {}; absorbed {}",
        f(&t["all_precision"]), f(&t["all_recall"]), f(&t["all_f1"]), t["impure"], t["absorbed"]
    );
    if let Some(list) = t["listing"].as_array() {
        for x in list {
            let fillers = x["fillers"].as_u64().unwrap_or(0);
            let extra = if fillers > 0 { format!(" + {fillers} distractors") } else { String::new() };
            println!("   - {:<28} {}{}", x["name"].as_str().unwrap_or(""), x["members"], extra);
        }
    }
    if let Some(d) = t["split_diagnostics"].as_array() {
        for x in d {
            println!(
                "   split {}: neighbours {} | best topic {}",
                x["item"], x["neighbours"], x["best_topic"]
            );
        }
    }
    println!("capture guard accuracy: {:.3}", f(&r["capture"]["accuracy"]));
    if let Some(fails) = r["capture"]["failures"].as_array() {
        for x in fails {
            println!("   x {} (expected capture={})", x["text"], x["expected_capture"]);
        }
    }
    println!(
        "\n{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8}",
        "pipeline", "recall", "mrr", "ndcg", "hit@1", "prec", "p50 ms", "p95 ms"
    );
    if let Some(p) = r["pipelines"].as_object() {
        for (name, m) in p {
            println!(
                "{:<10} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>8.2} {:>8.2}",
                name, f(&m["recall"]), f(&m["mrr"]), f(&m["ndcg"]), f(&m["hit1"]),
                f(&m["precision"]), f(&m["p50_ms"]), f(&m["p95_ms"])
            );
        }
    }
    println!("\nper-query (retrieve pipeline, misses flagged):");
    if let Some(pq) = r["per_query"].as_array() {
        for q in pq.iter().filter(|q| q["pipeline"] == "retrieve") {
            let flag = if f(&q["recall"]) < 1.0 || f(&q["hit1"]) < 1.0 { "!" } else { " " };
            println!(
                "{} {} R {:.2} MRR {:.2} P {:.2}  {}",
                flag, q["id"].as_str().unwrap_or(""), f(&q["recall"]), f(&q["mrr"]),
                f(&q["precision"]), q["query"].as_str().unwrap_or("")
            );
            if flag == "!" {
                println!("      top: {}", q["top"]);
            }
        }
    }
}

fn print_compare(prev: &Value, cur: &Value) {
    println!("\n=== delta vs {} ===", prev["label"]);
    for pipe in ["search", "retrieve"] {
        let (a, b) = (&prev["pipelines"][pipe], &cur["pipelines"][pipe]);
        println!(
            "{:<10} recall {:+.3}  mrr {:+.3}  ndcg {:+.3}  hit@1 {:+.3}  prec {:+.3}  p50 {:+.2} ms",
            pipe,
            f(&b["recall"]) - f(&a["recall"]),
            f(&b["mrr"]) - f(&a["mrr"]),
            f(&b["ndcg"]) - f(&a["ndcg"]),
            f(&b["hit1"]) - f(&a["hit1"]),
            f(&b["precision"]) - f(&a["precision"]),
            f(&b["p50_ms"]) - f(&a["p50_ms"]),
        );
    }
    println!(
        "topics     pair F1 {:+.3}   capture acc {:+.3}   chunks/s {:+.1}",
        f(&cur["topics"]["pair_f1"]) - f(&prev["topics"]["pair_f1"]),
        f(&cur["capture"]["accuracy"]) - f(&prev["capture"]["accuracy"]),
        f(&cur["ingest"]["chunks_per_sec"]) - f(&prev["ingest"]["chunks_per_sec"]),
    );
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
