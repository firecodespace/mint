//! Document ingestion helpers: parse bytes to text by file type, and chunk text
//! into embeddable pieces. Runs fully locally.

use anyhow::{anyhow, Result};

const TEXT_EXTS: &[&str] = &[
    "txt", "md", "markdown", "csv", "tsv", "log", "rs", "py", "js", "ts", "tsx", "jsx", "json",
    "html", "htm", "css", "scss", "toml", "yaml", "yml", "sh", "bash", "java", "kt", "c", "cpp",
    "cc", "h", "hpp", "go", "rb", "php", "sql", "xml", "ini", "cfg", "conf", "text",
];

fn ext_of(name: &str) -> String {
    name.rsplit('.').next().unwrap_or("").to_lowercase()
}

/// Parse raw file bytes into plain text. Supports UTF-8 text/code formats and PDF.
pub fn parse(filename: &str, bytes: &[u8]) -> Result<String> {
    let ext = ext_of(filename);
    if ext == "pdf" {
        // pdf-extract can panic on PDFs with unusual encodings; catch it so a
        // difficult file yields a clean error rather than unwinding the task.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pdf_extract::extract_text_from_mem(bytes)
        }));
        return match result {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(e)) => Err(anyhow!("failed to parse PDF: {e}")),
            Err(_) => Err(anyhow!(
                "this PDF uses an encoding the local parser can't read; try exporting it to text"
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
