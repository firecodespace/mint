//! Document ingestion helpers: parse bytes to text by file type, and chunk text
//! into embeddable pieces. Runs fully locally.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use pdfium_render::prelude::Pdfium;

const TEXT_EXTS: &[&str] = &[
    "txt", "md", "markdown", "csv", "tsv", "log", "rs", "py", "js", "ts", "tsx", "jsx", "json",
    "html", "htm", "css", "scss", "toml", "yaml", "yml", "sh", "bash", "java", "kt", "c", "cpp",
    "cc", "h", "hpp", "go", "rb", "php", "sql", "xml", "ini", "cfg", "conf", "text",
];

fn ext_of(name: &str) -> String {
    name.rsplit('.').next().unwrap_or("").to_lowercase()
}

/// Locate and bind the pdfium native library, if available.
fn bind_pdfium() -> Option<Pdfium> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("MINT_PDFIUM_PATH") {
        if !p.trim().is_empty() {
            candidates.push(PathBuf::from(p));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(Pdfium::pdfium_platform_library_name()));
        }
    }
    candidates.push(Pdfium::pdfium_platform_library_name_at_path("./"));
    candidates.push(Pdfium::pdfium_platform_library_name_at_path("./lib/"));

    for c in candidates {
        if let Ok(bindings) = Pdfium::bind_to_library(&c) {
            return Some(Pdfium::new(bindings));
        }
    }
    Pdfium::bind_to_system_library().ok().map(Pdfium::new)
}

/// Extract text from a PDF with pdfium (robust; Google's engine).
fn parse_pdf_pdfium(bytes: &[u8]) -> Option<String> {
    let pdfium = bind_pdfium()?;
    let doc = pdfium.load_pdf_from_byte_slice(bytes, None).ok()?;
    let mut text = String::new();
    for page in doc.pages().iter() {
        if let Ok(t) = page.text() {
            text.push_str(&t.all());
            text.push('\n');
        }
    }
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Parse raw file bytes into plain text. Supports UTF-8 text/code formats and PDF.
pub fn parse(filename: &str, bytes: &[u8]) -> Result<String> {
    let ext = ext_of(filename);
    if ext == "pdf" {
        // Prefer pdfium (handles almost any PDF). Fall back to the pure-Rust
        // pdf-extract, guarded against its panics on unusual encodings.
        if let Some(text) = parse_pdf_pdfium(bytes) {
            return Ok(text);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pdf_extract::extract_text_from_mem(bytes)
        }));
        return match result {
            Ok(Ok(text)) if !text.trim().is_empty() => Ok(text),
            Ok(Ok(_)) => Err(anyhow!("no extractable text found in this PDF (it may be scanned images)")),
            Ok(Err(e)) => Err(anyhow!("failed to parse PDF: {e}")),
            Err(_) => Err(anyhow!(
                "this PDF could not be parsed locally (pdfium unavailable and the fallback parser failed)"
            )),
        };
    }
    if TEXT_EXTS.contains(&ext.as_str()) || ext.is_empty() {
        return Ok(String::from_utf8_lossy(bytes).to_string());
    }
    // Fall back to UTF-8 if it decodes cleanly; otherwise it is unsupported.
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Err(anyhow!(
            "unsupported file type '.{ext}' (supported: text/code formats and PDF)"
        )),
    }
}

const TARGET: usize = 900;
const MAX: usize = 1300;
const OVERLAP: usize = 120;

/// Chunk text semantically: split into structural blocks (paragraphs, with
/// markdown headings starting a new block), pack blocks to ~TARGET chars, and
/// split any oversized block on sentence boundaries. Keeps chunks aligned to
/// meaning, which improves retrieval quality.
pub fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for block in split_blocks(text) {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        if block.chars().count() > MAX {
            if !current.trim().is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            chunks.extend(pack_sentences(block));
            continue;
        }
        if current.chars().count() + block.chars().count() > TARGET && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(block);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks.into_iter().filter(|c| !c.trim().is_empty()).collect()
}

/// Split text into structural blocks: blank lines separate paragraphs, and a
/// markdown heading (or an ALL-CAPS/`:`-terminated line) starts a new block.
fn split_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cur = String::new();
    for line in text.split('\n') {
        if line.trim().is_empty() {
            if !cur.trim().is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let t = line.trim_start();
        let is_heading = t.starts_with('#')
            || (t.chars().count() < 80 && t.ends_with(':') && !t.contains(". "));
        if is_heading && !cur.trim().is_empty() {
            blocks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }
    if !cur.trim().is_empty() {
        blocks.push(cur);
    }
    blocks
}

/// Pack a large block into ~TARGET-char chunks on sentence boundaries.
fn pack_sentences(block: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for sentence in split_sentences(block) {
        if cur.chars().count() + sentence.chars().count() > TARGET && !cur.is_empty() {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(sentence.trim());
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Split on sentence-ending punctuation; hard-split any monster sentence.
fn split_sentences(s: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') && cur.trim().chars().count() > 24 {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur);
    }
    let mut out = Vec::new();
    for sent in sentences {
        if sent.chars().count() > MAX {
            out.extend(hard_split(&sent));
        } else {
            out.push(sent);
        }
    }
    out
}

fn hard_split(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + TARGET).min(chars.len());
        out.push(chars[start..end].iter().collect::<String>());
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(OVERLAP);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{chunk_text, MAX};

    #[test]
    fn chunks_respect_max_and_keep_content() {
        let para = "Sentence about transit photometry and stellar variability. ".repeat(40);
        let text = format!("# Heading\n\n{para}\n\n## Second\n\nShort tail paragraph.");
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 2, "long text should split");
        for c in &chunks {
            assert!(c.chars().count() <= MAX, "chunk over MAX: {}", c.chars().count());
            assert!(!c.trim().is_empty());
        }
        let joined = chunks.join(" ");
        assert!(joined.contains("Short tail paragraph"));
        assert!(joined.contains("# Heading"));
    }

    #[test]
    fn small_text_is_one_chunk() {
        assert_eq!(chunk_text("just one line").len(), 1);
        assert!(chunk_text("   \n\n  ").is_empty());
    }

    #[test]
    fn unbroken_text_is_hard_split() {
        let blob = "x".repeat(5000);
        let chunks = chunk_text(&blob);
        assert!(chunks.len() >= 5);
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX));
    }
}
