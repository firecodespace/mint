//! Labeled benchmark corpus: documents + notes across several subjects (with
//! distractors), and queries with graded relevance judgments. Modeled on real
//! usage: research papers, a resume, a study guide, personal goals.

pub struct Doc {
    pub key: &'static str,
    pub title: &'static str,
    pub topic: &'static str,
    pub text: &'static str,
}

pub struct Note {
    pub key: &'static str,
    pub topic: &'static str,
    pub text: &'static str,
}

/// A query and its relevant items as (item key, grade). Grade 2 = primary
/// answer source, 1 = useful supporting source. Unlisted items are irrelevant.
pub struct Query {
    pub id: &'static str,
    pub text: &'static str,
    pub relevant: &'static [(&'static str, u8)],
}

/// A chat message and whether it should be captured as a durable memory.
pub struct CaptureCase {
    pub text: &'static str,
    pub capture: bool,
}

pub const DOCS: &[Doc] = &[
    Doc {
        key: "exo_doc",
        title: "Physics-Informed Transit Detection.pdf",
        topic: "exoplanet",
        text: "# Physics-Informed Transit Detection for Exoplanets

## Abstract
We present a physics-informed neural network for detecting exoplanet transits in \
Kepler and TESS light curves. A transit is a periodic, box-shaped dip in stellar \
brightness caused by a planet crossing the face of its host star. Classical pipelines \
such as Box Least Squares search for these dips directly, but they struggle with \
stellar variability, instrumental systematics, and eclipsing binaries that mimic \
planetary signals.

## Method
Our model couples a one-dimensional convolutional classifier with a differentiable \
transit model (limb-darkened Mandel-Agol shapes). Before classification, each light \
curve is detrended with a Gaussian process whose kernel captures quasi-periodic stellar \
rotation. The classifier receives the phase-folded global view and a local view \
centered on the candidate event. The physics prior constrains transit depth, duration, \
and ingress to be consistent with Keplerian orbits, which suppresses false positives.

## Results
On the Kepler DR25 threshold-crossing events the model reaches 0.94 precision and 0.89 \
recall on confirmed planets. Most remaining false positives are grazing eclipsing \
binaries with secondary eclipses below the noise floor. Ablations show the Gaussian \
process detrending contributes the largest single gain.

## Future work
We plan to extend the approach to TESS full-frame images and to estimate planet radius \
posteriors with Bayesian inference and MCMC sampling.",
    },
    Doc {
        key: "coresum_doc",
        title: "coresum.pdf",
        topic: "coresum",
        text: "# CoreSum: Budgeted Video Summarization

## Overview
CoreSum selects a compact set of keyframes and shots that summarize a long video \
under a fixed token budget. Frames are embedded with a vision encoder, grouped into \
shots, and scored for importance. A grid-budget optimizer then allocates the budget \
across temporal segments so that every part of the story is represented.

## Scoring
Shot importance is predicted by a Ridge regression scorer trained on human annotations. \
Features include visual novelty, motion energy, caption salience, and audio peaks. The \
Ridge model was chosen for speed and stability over a larger transformer head.

## Evaluation
We report Temporal Coverage Score (TCS), which measures how well the summary covers the \
annotated key events over time. The evaluation harness is run_eval.py, and evaluate.py \
computes TCS per video and averages across the benchmark split. The current system \
reaches a TCS of 0.585 against a target of 0.81, a gap of 0.225. The gap is concentrated \
in videos longer than twenty minutes, where the grid budget spreads too thin.

## Performance
End-to-end inference takes between 15 and 47 seconds per video on a laptop GPU. The \
vision encoder dominates runtime; caching frame embeddings would roughly halve it.

## Next steps
Tune the grid budget for long videos, add a coverage penalty to the scorer, and rerun \
the full benchmark before the next milestone.",
    },
    Doc {
        key: "usa_doc",
        title: "Usa_StudyGuide.pdf",
        topic: "usa",
        text: "# Studying in the United States: A Practical Guide

## Student visas
Most international students hold an F-1 visa. To maintain F-1 status you must stay \
enrolled full time, keep your passport and Form I-20 valid, and report changes of \
address to your Designated School Official (DSO), who updates your record in SEVIS, the \
Student and Exchange Visitor Information System.

## Working while studying
On-campus employment is allowed up to 20 hours per week during the semester. \
Off-campus work requires authorization. Curricular Practical Training (CPT) allows paid \
internships that are an integral part of your curriculum; it is authorized by your DSO \
and tied to a specific employer and dates. Unauthorized employment is a serious \
violation of F-1 status.

## Optional Practical Training
Optional Practical Training (OPT) provides up to twelve months of work authorization in \
your field of study, and STEM graduates may qualify for a 24-month extension. You can \
apply for post-completion OPT up to 90 days before your program end date by filing Form \
I-765 with USCIS after your DSO recommends it in SEVIS.

## Internships and careers
Internships help students gain practical experience and improve employability. Career \
centers host job fairs, review resumes, and run mock interviews. Start searching early, \
tailor your resume to each role, and confirm CPT eligibility before accepting an offer.

## Health and housing
Most universities require health insurance. Arrange housing early, read leases \
carefully, and budget for deposits and utilities.",
    },
    Doc {
        key: "resume_doc",
        title: "Resume (Aug 2026) (1).pdf",
        topic: "profile",
        text: "Yash Bendresh
Computer Science and Engineering student | Founder, Xarch Labs

EDUCATION
B.Tech in Computer Science and Engineering, expected 2027. Coursework: machine \
learning, distributed systems, computer networks, IoT-based product design, databases.

EXPERIENCE
Founder, Xarch Labs (2025 - present). Building local-first AI products. Led design of an \
offline memory engine using vector search and on-device language models. Managed a small \
team and shipped a desktop application.

PROJECTS
Physics-informed AI for exoplanet detection: combined Gaussian processes, Bayesian \
inference, and MCMC with a convolutional classifier on Kepler light curves.
AEGIS: a zero-trust cloud security system with mutual TLS, policy-based access control, \
and anomaly detection on audit logs.
Hierarchical cognitive memory architecture: hybrid retrieval and context-aware reasoning \
for AI assistants.

SKILLS
Python, PyTorch, Rust, TypeScript, React, SQL, Docker, Linux. Machine learning, \
Bayesian modeling, computer vision, security engineering, system design.

ACHIEVEMENTS
Published a workshop paper on memory architectures. Led a winning hackathon team.",
    },
    Doc {
        key: "hcma_doc",
        title: "Hierarchical Cognitive Memory Architecture for Hybrid Retrieval.pdf",
        topic: "hcma",
        text: "# Hierarchical Cognitive Memory Architecture (HCMA)

## Motivation
Large language model assistants forget between sessions and retrieve context poorly. \
HCMA organizes memory into layers inspired by human cognition so that an assistant can \
store, consolidate, and recall information over long horizons.

## Memory layers
Episodic memory stores raw experiences such as conversation turns with timestamps. \
Semantic memory stores distilled facts and concepts abstracted from many episodes. \
Procedural memory stores how-to knowledge and recurring patterns.

## Consolidation
Consolidation periodically promotes frequently accessed episodic memories into semantic \
memory. A summarizer merges related episodes, resolves contradictions, and assigns an \
understanding score. Low-salience memories decay and are archived rather than deleted.

## Hybrid retrieval
Queries are answered with hybrid retrieval that fuses dense semantic similarity with \
sparse keyword matching using reciprocal rank fusion, followed by a salience-aware \
reranker that weighs recency, access frequency, and graph connectivity.

## Evaluation
On a long-horizon dialogue benchmark HCMA improves answer accuracy over a flat vector \
store while using fewer context tokens.",
    },
    Doc {
        key: "cooking_doc",
        title: "Weeknight Recipes.txt",
        topic: "cooking",
        text: "# Weeknight Recipes

## Lemon garlic pasta
Boil spaghetti in salted water. Meanwhile saute sliced garlic in olive oil, add lemon \
zest and juice, then toss with the pasta, a handful of spinach, and grated parmesan.

## Sheet pan chicken
Toss chicken thighs, potatoes, and carrots with oil, paprika, salt, and pepper. Roast at \
220 C for 35 minutes, turning once.

## Chickpea curry
Cook onion, ginger, and garlic, add curry powder and tomatoes, then simmer chickpeas in \
coconut milk for twenty minutes. Finish with lime and cilantro.",
    },
];

pub const NOTES: &[Note] = &[
    Note { key: "n_exo1", topic: "exoplanet", text: "Tried a 1D CNN on Kepler light curves; recall on confirmed planets was 0.87 but eclipsing binaries cause many false positives." },
    Note { key: "n_exo2", topic: "exoplanet", text: "Decided to denoise the light curves with a Gaussian process before classification to remove stellar variability." },
    Note { key: "n_exo3", topic: "exoplanet", text: "I want to submit the exoplanet transit paper to the NeurIPS ML4PS workshop." },
    Note { key: "n_cs1", topic: "coresum", text: "CoreSum TCS is stuck at 0.585 while the target is 0.81; the gap is mostly on long videos." },
    Note { key: "n_cs2", topic: "coresum", text: "CoreSum benchmarks run through run_eval.py, and evaluate.py computes TCS per video." },
    Note { key: "n_cs3", topic: "coresum", text: "CoreSum inference takes 15 to 47 seconds per video on the laptop GPU, too slow for the demo." },
    Note { key: "n_usa1", topic: "usa", text: "My I-20 expires in May, so I need to request an extension from the DSO." },
    Note { key: "n_p1", topic: "piano", text: "I want to learn piano by the end of the year." },
    Note { key: "n_p2", topic: "piano", text: "Practicing piano scales for 20 minutes every day on the grand piano at the community hall." },
    Note { key: "n_p3", topic: "piano", text: "Bought a Yamaha P-45 digital keyboard so I can practice piano at home." },
    Note { key: "n_h1", topic: "hcma", text: "HCMA consolidation should promote episodic memories to semantic memory after repeated access." },
    Note { key: "n_f1", topic: "fitness", text: "Running 5k three times a week; my best time so far is 26 minutes." },
    Note { key: "n_f2", topic: "fitness", text: "Deadlift is up to 100 kg, focusing on form over weight." },
    Note { key: "n_c1", topic: "cooking", text: "My go-to dinner is lemon garlic pasta with spinach." },
];

pub const QUERIES: &[Query] = &[
    Query { id: "q01", text: "what type of internships are best for me?", relevant: &[("resume_doc", 2), ("usa_doc", 1)] },
    Query { id: "q02", text: "check from my resume what skills I have", relevant: &[("resume_doc", 2)] },
    Query { id: "q03", text: "what is the TCS gap in our coresum benchmarks?", relevant: &[("n_cs1", 2), ("coresum_doc", 2), ("n_cs2", 1)] },
    Query { id: "q04", text: "set a deadline for bringing our benchmarks for core-sum by the end of this week", relevant: &[("coresum_doc", 2), ("n_cs2", 2), ("n_cs1", 1), ("n_cs3", 1)] },
    Query { id: "q05", text: "summarize my exoplanet research", relevant: &[("exo_doc", 2), ("n_exo1", 2), ("n_exo2", 2), ("n_exo3", 1)] },
    Query { id: "q06", text: "how do I keep my F-1 status while working an internship?", relevant: &[("usa_doc", 2)] },
    Query { id: "q07", text: "what did I decide about piano practice?", relevant: &[("n_p2", 2), ("n_p3", 1), ("n_p1", 1)] },
    Query { id: "q08", text: "how does consolidation work in HCMA?", relevant: &[("hcma_doc", 2), ("n_h1", 2)] },
    Query { id: "q09", text: "SEVIS", relevant: &[("usa_doc", 2)] },
    Query { id: "q10", text: "Kepler light curve denoising", relevant: &[("n_exo2", 2), ("exo_doc", 2), ("n_exo1", 1)] },
    Query { id: "q11", text: "what are my goals for this year?", relevant: &[("n_p1", 2), ("n_exo3", 1)] },
    Query { id: "q12", text: "Ridge regression scorer grid budget", relevant: &[("coresum_doc", 2)] },
    Query { id: "q13", text: "what is my coresum TCS score right now?", relevant: &[("n_cs1", 2), ("coresum_doc", 1)] },
    Query { id: "q14", text: "what are my strengths as an engineer?", relevant: &[("resume_doc", 2)] },
    Query { id: "q15", text: "how slow is coresum inference?", relevant: &[("n_cs3", 2), ("coresum_doc", 1)] },
    Query { id: "q16", text: "what keyboard did I buy?", relevant: &[("n_p3", 2)] },
    Query { id: "q17", text: "how fast is my 5k?", relevant: &[("n_f1", 2)] },
    Query { id: "q18", text: "what is OPT and when can I apply?", relevant: &[("usa_doc", 2)] },
    Query { id: "q19", text: "where should I submit my exoplanet paper?", relevant: &[("n_exo3", 2), ("exo_doc", 1)] },
    Query { id: "q20", text: "episodic versus semantic memory", relevant: &[("hcma_doc", 2), ("n_h1", 1)] },
];

/// Distractor domains for scale testing. The first three are deliberate HARD
/// negatives (vocabulary near the real subjects: astronomy, video, travel) but
/// never answer a labeled query; the rest are unrelated noise.
const FILLER_DOMAINS: &[(&str, &[&str], &[&str])] = &[
    ("astronomy", &["galaxy", "telescope", "nebula", "star cluster", "comet", "observatory", "spiral arm"],
     &["was photographed through a backyard {x}", "appears brighter during the winter season",
       "was catalogued by amateur astronomers in the nineteenth century", "is visible without optics from dark sites",
       "inspired a new generation of {x} hobbyists"]),
    ("video", &["camera rig", "color grade", "timeline edit", "drone shot", "lens flare", "b-roll", "frame rate"],
     &["was recorded for the travel vlog", "needed a warmer {x} in post", "was cut for the wedding film",
       "looked cinematic at twenty-four frames", "was stabilized on a gimbal before the {x}"]),
    ("travel", &["itinerary", "passport photo", "hostel", "rail pass", "packing list", "museum ticket", "ferry"],
     &["was booked for the summer holiday", "saved money compared with a hotel", "covered five cities in ten days",
       "needed a {x} checked the night before", "was recommended by a friend who lived abroad"]),
    ("gardening", &["tomato bed", "compost bin", "rose bush", "drip line", "seed tray", "raised bed", "mulch"],
     &["needs watering every morning in summer", "was pruned back after flowering", "attracted bees to the {x}",
       "grew faster with a layer of {x}", "was moved to the sunny side of the yard"]),
    ("finance", &["index fund", "tax return", "budget sheet", "savings goal", "credit card", "mortgage", "invoice"],
     &["was reviewed at the end of the quarter", "reduced monthly spending on dining out", "was filed before the deadline",
       "tracks every expense in a {x}", "was rebalanced to lower fees"]),
    ("carpentry", &["oak table", "dovetail joint", "workbench", "chisel set", "sanding block", "wood glue", "clamp"],
     &["was finished with two coats of oil", "needed a sharper {x} for clean cuts", "was built from reclaimed timber",
       "held tight overnight with a {x}", "was measured twice before cutting"]),
    ("maritime", &["lighthouse", "schooner", "harbor", "sea chart", "anchor", "tide table", "shipwreck"],
     &["guided sailors along the rocky coast", "was restored by the local historical society", "appeared on an old {x}",
       "survived a storm in the eighteenth century", "is now a small museum by the {x}"]),
];

/// Deterministic pseudo-random generator (LCG) so scale runs are reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
    fn pick<'a>(&mut self, xs: &'a [&'a str]) -> &'a str {
        xs[self.next() % xs.len()]
    }
}

/// Generate `n` distractor documents: (title, text, true topic). Each is a few
/// paragraphs of domain sentences, enough to produce several chunks.
pub fn filler_docs(n: usize) -> Vec<(String, String, &'static str)> {
    let mut rng = Lcg(0x5eed_2026);
    (0..n)
        .map(|i| {
            let (domain, nouns, preds) = FILLER_DOMAINS[i % FILLER_DOMAINS.len()];
            let mut text = format!("# Notes on {domain} {i}\n\n");
            for _ in 0..4 {
                for _ in 0..6 {
                    let noun = rng.pick(nouns);
                    let pred = rng.pick(preds).replace("{x}", rng.pick(nouns));
                    text.push_str(&format!("The {noun} {pred}. "));
                }
                text.push_str("\n\n");
            }
            (format!("{domain}-notes-{i}.txt"), text, domain)
        })
        .collect()
}

/// One long document (for the ingestion A/B micro-benchmark).
pub fn long_doc(paragraphs: usize) -> String {
    let mut rng = Lcg(0xab_1e);
    let (_, nouns, preds) = FILLER_DOMAINS[3];
    let mut text = String::new();
    for _ in 0..paragraphs {
        for _ in 0..7 {
            let pred = rng.pick(preds).replace("{x}", rng.pick(nouns));
            text.push_str(&format!("The {} {}. ", rng.pick(nouns), pred));
        }
        text.push_str("\n\n");
    }
    text
}

/// Sync-policy case: should this text stay on the device, and why.
pub struct PolicyCase {
    pub text: &'static str,
    /// Expected category when it must stay local; None = may sync.
    pub local: Option<&'static str>,
}

pub const POLICY_CASES: &[PolicyCase] = &[
    PolicyCase { text: "my wifi password is hunter2", local: Some("secret") },
    PolicyCase { text: "OpenAI key sk-proj-A1b2C3d4E5f6G7h8I9j0", local: Some("secret") },
    PolicyCase { text: "aws access key AKIAIOSFODNN7EXAMPLE", local: Some("secret") },
    PolicyCase { text: "the building door pin is 4821", local: Some("secret") },
    PolicyCase { text: "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123signature", local: Some("secret") },
    PolicyCase { text: "card 4111 1111 1111 1111 exp 09/28", local: Some("financial") },
    PolicyCase { text: "salary goes to account number 004512349876", local: Some("financial") },
    PolicyCase { text: "IBAN DE89370400440532013000 for the rent", local: Some("financial") },
    PolicyCase { text: "my SSN is 123-45-6789", local: Some("government_id") },
    PolicyCase { text: "aadhaar 1234 5678 9012 for KYC", local: Some("government_id") },
    PolicyCase { text: "PAN ABCDE1234F", local: Some("government_id") },
    PolicyCase { text: "passport number K1234567 expires 2031", local: Some("government_id") },
    PolicyCase { text: "I'm allergic to peanuts", local: Some("health") },
    PolicyCase { text: "I was diagnosed with asthma last year", local: Some("health") },
    PolicyCase { text: "my therapist moved our session to Thursday", local: Some("health") },
    PolicyCase { text: "I take 50mg of medication every morning", local: Some("health") },
    PolicyCase { text: "email me at yash@example.com", local: Some("contact") },
    PolicyCase { text: "call the landlord at +91 98765 43210", local: Some("contact") },
    PolicyCase { text: "my address is 12 MG Road, Pune", local: Some("contact") },
    PolicyCase { text: "reset my password tomorrow", local: None },
    PolicyCase { text: "the password is required for the admin panel", local: None },
    PolicyCase { text: "Cancer detection with CNNs reaches 0.94 AUC on the benchmark", local: None },
    PolicyCase { text: "The patient cohort in the study showed fewer symptoms", local: None },
    PolicyCase { text: "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed the parser", local: None },
    PolicyCase { text: "Meeting at 10:30 on 2026-10-02 in room 204", local: None },
    PolicyCase { text: "see https://example.com/docs/AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", local: None },
    PolicyCase { text: "version 2026.10.02 shipped to 1200 users", local: None },
    PolicyCase { text: "CoreSum inference takes 15 to 47 seconds per video", local: None },
    PolicyCase { text: "Our team switched from Postgres to SQLite for the edge build", local: None },
    PolicyCase { text: "I decided to use Rust for the backend", local: None },
    PolicyCase { text: "My exam is on Friday", local: None },
    PolicyCase { text: "Bought a Yamaha P-45 keyboard for 45000 rupees", local: None },
    PolicyCase { text: "Order 1234567 shipped yesterday", local: None },
    PolicyCase { text: "The API key rotation policy is every 90 days", local: None },
];

/// Version-chain case: earlier memories, then a new one; which earlier memory
/// (if any) the new one updates, and optionally a question whose answer must
/// come from the CURRENT version.
pub struct VersionCase {
    pub id: &'static str,
    pub old: &'static [&'static str],
    pub new: &'static str,
    pub updates: Option<usize>,
    pub query: Option<&'static str>,
}

pub const VERSION_CASES: &[VersionCase] = &[
    VersionCase { id: "v01", old: &["My exam is on Friday"], new: "My exam got moved to Monday", updates: Some(0), query: Some("when is my exam?") },
    VersionCase { id: "v02", old: &["I prefer dark roast coffee"], new: "I switched to green tea and stopped drinking coffee", updates: Some(0), query: Some("what do I like to drink?") },
    VersionCase { id: "v03", old: &["CoreSum TCS is 0.585 on the benchmark"], new: "CoreSum TCS improved to 0.71 after tuning the grid budget", updates: Some(0), query: Some("what is the CoreSum TCS?") },
    VersionCase { id: "v04", old: &["Our edge build uses Postgres for storage"], new: "We moved the edge build from Postgres to SQLite", updates: Some(0), query: Some("which database does the edge build use?") },
    VersionCase { id: "v05", old: &["The DSO meeting is on Tuesday at 3pm"], new: "The DSO meeting was rescheduled to Thursday at 11am", updates: Some(0), query: Some("when is the DSO meeting?") },
    VersionCase { id: "v06", old: &["I live in Pune"], new: "I moved to Bangalore last month", updates: Some(0), query: Some("which city do I live in?") },
    VersionCase { id: "v07", old: &["The CoreSum report deadline is October 2"], new: "The CoreSum report deadline is now October 9", updates: Some(0), query: Some("when is the CoreSum report due?") },
    VersionCase { id: "v08", old: &["My exam is on Friday"], new: "My brother's exam is on Monday", updates: None, query: None },
    VersionCase { id: "v09", old: &["I prefer dark roast coffee"], new: "I bought a new burr grinder for coffee", updates: None, query: None },
    VersionCase { id: "v10", old: &["CoreSum TCS is 0.585 on the benchmark"], new: "CoreSum inference takes 15 seconds per video", updates: None, query: None },
    VersionCase { id: "v11", old: &["Practicing piano scales every day"], new: "Bought a Yamaha P-45 keyboard", updates: None, query: None },
    VersionCase { id: "v12", old: &["Running 5k three times a week"], new: "Deadlift is up to 100 kg now", updates: None, query: None },
    VersionCase { id: "v13", old: &["I want to learn piano by the end of the year"], new: "My goal is still to learn piano by the end of the year", updates: None, query: None },
    VersionCase { id: "v14", old: &["My exam is on Friday", "My brother's exam is on Monday"], new: "My exam was moved to Wednesday", updates: Some(0), query: Some("when is my exam?") },
];

pub const CAPTURE_CASES: &[CaptureCase] = &[
    // Durable statements: should be captured.
    CaptureCase { text: "My name is Yash", capture: true },
    CaptureCase { text: "I work at Xarch Labs as the founder", capture: true },
    CaptureCase { text: "I prefer dark roast coffee", capture: true },
    CaptureCase { text: "I decided to use Rust for the backend", capture: true },
    CaptureCase { text: "My exam is on Friday", capture: true },
    CaptureCase { text: "I'm allergic to peanuts", capture: true },
    CaptureCase { text: "We moved the launch to October", capture: true },
    CaptureCase { text: "The CoreSum TCS target is 0.81", capture: true },
    CaptureCase { text: "Started learning piano this week", capture: true },
    CaptureCase { text: "Our team switched from Postgres to SQLite for the edge build", capture: true },
    // Questions, requests, commands, small talk: must NOT be captured.
    CaptureCase { text: "check from my resume", capture: false },
    CaptureCase { text: "what type of internships are best for me?", capture: false },
    CaptureCase { text: "tell me about my projects", capture: false },
    CaptureCase { text: "how does HCMA work", capture: false },
    CaptureCase { text: "summarize my exoplanet research", capture: false },
    CaptureCase { text: "hi", capture: false },
    CaptureCase { text: "thanks", capture: false },
    CaptureCase { text: "can you set a deadline for friday", capture: false },
    CaptureCase { text: "show my tasks", capture: false },
    CaptureCase { text: "What's my name?", capture: false },
    CaptureCase { text: "explain OPT", capture: false },
    CaptureCase { text: "set a deadline for bringing our benchmarks for core-sum by the end of this week", capture: false },
    CaptureCase { text: "schedule a call with the DSO next Tuesday", capture: false },
    CaptureCase { text: "compare CoreSum with the baseline", capture: false },
];
