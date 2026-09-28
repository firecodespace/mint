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

/// Chunk text into ~900-char pieces on paragraph boundaries, hard-splitting any
/// oversized paragraph with a small overlap.
pub fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.chars().count() > MAX {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            chunks.extend(hard_split(para));
            continue;
        }
        if current.chars().count() + para.chars().count() > TARGET && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks.into_iter().filter(|c| !c.trim().is_empty()).collect()
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
