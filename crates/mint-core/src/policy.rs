//! Sync policy: decides, per memory, whether it may leave the device.
//!
//! Deterministic and local (no LLM, no network): it runs on every write and
//! again at sync time. Anything sensitive stays on the device with a stated
//! reason; everything else is eligible to sync.
//!
//! Kept local:
//!   - secret         credentials, API keys, tokens, private keys, passwords
//!   - financial      payment card numbers (Luhn-checked), bank/IBAN numbers
//!   - government_id  SSN, Aadhaar, PAN, passport numbers
//!   - health         personal health facts (first-person context only, so a
//!                    research paper about cancer detection still syncs)
//!   - contact        email addresses, phone numbers, home addresses
//! Plus two non-content rules applied by the engine: user choice, and derived
//! data (entity hubs) that each device can recompute.

/// Why a piece of text must stay on the device.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub category: &'static str,
    pub reason: String,
}

fn finding(category: &'static str, what: &str) -> Option<Finding> {
    Some(Finding {
        category,
        reason: format!("Stays on device: {what}"),
    })
}

/// Scan text for content that must not leave the device. First match wins,
/// most severe categories first.
pub fn scan(text: &str) -> Option<Finding> {
    secret(text)
        .or_else(|| financial(text))
        .or_else(|| government_id(text))
        .or_else(|| health(text))
        .or_else(|| contact(text))
}

// ---- tokenization helpers -------------------------------------------------

/// Whitespace tokens with surrounding punctuation trimmed.
fn tokens(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| {
                matches!(c, ',' | ';' | '"' | '\'' | '(' | ')' | '[' | ']' | '<' | '>' | '`')
                    || (c == '.' || c == ':' || c == '!' || c == '?')
            })
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Lowercase words (alphanumeric runs).
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// Runs of digits allowing common separators (space, dash, dot, parens, +),
/// returned as (digits only, raw run). Used for card / ID / phone detection.
/// Runs glued to letters ("Yz0123456789", "v2", "abc123") are part of an
/// identifier, not a number, and are skipped.
fn digit_runs(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut raw = String::new();
    let mut digits = String::new();
    let mut glued = false;
    let chars: Vec<char> = text.chars().collect();
    let mut flush = |raw: &mut String, digits: &mut String, glued: &mut bool, next: Option<char>| {
        let glued_after = next.map(|c| c.is_alphabetic()).unwrap_or(false);
        if !digits.is_empty() && !*glued && !glued_after {
            out.push((digits.clone(), raw.trim().to_string()));
        }
        raw.clear();
        digits.clear();
        *glued = false;
    };
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_digit() {
            if digits.is_empty() && raw.is_empty() {
                glued = i > 0 && chars[i - 1].is_alphabetic();
            }
            raw.push(c);
            digits.push(c);
        } else if !digits.is_empty()
            && matches!(c, ' ' | '-' | '.' | '(' | ')')
            && chars.get(i + 1).map(|n| n.is_ascii_digit() || *n == '(' || *n == ' ').unwrap_or(false)
        {
            raw.push(c);
        } else if (c == '+' || c == '(') && digits.is_empty() {
            raw.push(c);
        } else {
            flush(&mut raw, &mut digits, &mut glued, Some(c));
        }
    }
    flush(&mut raw, &mut digits, &mut glued, None);
    out
}

fn luhn(digits: &str) -> bool {
    let mut sum = 0;
    for (i, c) in digits.chars().rev().enumerate() {
        let mut d = c.to_digit(10).unwrap_or(0);
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
    }
    sum % 10 == 0 && !digits.is_empty()
}

/// Does `keyword` appear followed (within a few characters) by "is", ":" or
/// "=" and then a value-like token? "my wifi password is hunter2" -> true,
/// "reset my password tomorrow" -> false. `need_digit` requires the value to
/// contain a digit (PINs, OTPs, CVVs).
fn keyword_value(lower: &str, keyword: &str, need_digit: bool) -> bool {
    const NOT_VALUES: &[&str] = &[
        "the", "a", "an", "my", "your", "not", "required", "changed", "reset", "expired",
        "weak", "strong", "wrong", "correct", "missing", "invalid", "same", "new", "old",
        "stored", "saved", "set", "empty", "too", "very", "in", "on", "at", "for", "to",
    ];
    let mut start = 0;
    while let Some(pos) = lower[start..].find(keyword) {
        let at = start + pos;
        let before_ok = at == 0 || !lower[..at].chars().last().unwrap_or(' ').is_alphanumeric();
        let after = &lower[at + keyword.len()..];
        let after_ok = !after.chars().next().map(|c| c.is_alphanumeric()).unwrap_or(false);
        if before_ok && after_ok {
            let rest = after.trim_start();
            let rest = rest
                .strip_prefix("is")
                .map(|r| r.trim_start())
                .or_else(|| rest.strip_prefix(':').map(|r| r.trim_start()))
                .or_else(|| rest.strip_prefix('=').map(|r| r.trim_start()));
            if let Some(r) = rest {
                let value: String = r
                    .chars()
                    .take_while(|c| !c.is_whitespace())
                    .collect::<String>()
                    .trim_matches(|c: char| matches!(c, '.' | ',' | '"' | '\'' | ';'))
                    .to_string();
                let has_digit = value.chars().any(|c| c.is_ascii_digit());
                if value.chars().count() >= 3
                    && !NOT_VALUES.contains(&value.as_str())
                    && (!need_digit || has_digit)
                {
                    return true;
                }
            }
        }
        start = at + keyword.len();
    }
    false
}

// ---- categories -------------------------------------------------------------

fn secret(text: &str) -> Option<Finding> {
    if text.contains("-----BEGIN") && text.contains("PRIVATE KEY") {
        return finding("secret", "contains a private key");
    }
    const PREFIXES: &[&str] = &[
        "sk-", "sk_live_", "sk_test_", "pk_live_", "rk_live_", "AKIA", "ASIA", "ghp_", "gho_",
        "ghs_", "ghu_", "github_pat_", "xoxb-", "xoxp-", "xapp-", "AIza", "glpat-", "hf_",
        "npm_",
    ];
    for t in tokens(text) {
        let n = t.chars().count();
        if n >= 16 && PREFIXES.iter().any(|p| t.starts_with(p)) {
            return finding("secret", "contains an API key or access token");
        }
        // JSON Web Token.
        if n >= 30 && t.starts_with("eyJ") && t.matches('.').count() == 2 {
            return finding("secret", "contains an authentication token");
        }
        // Long random-looking token: mixed case + digits, not a URL/path, not
        // a plain hex digest (commit SHAs and hashes are not secrets).
        if n >= 32
            && !t.contains('/')
            && !t.contains('@')
            && t.chars().any(|c| c.is_ascii_lowercase())
            && t.chars().any(|c| c.is_ascii_uppercase())
            && t.chars().any(|c| c.is_ascii_digit())
            && t.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '='))
        {
            return finding("secret", "contains a long random token (looks like a key)");
        }
    }
    let lower = text.to_lowercase();
    const CREDENTIALS: &[&str] = &[
        "password", "passcode", "passwd", "passphrase", "api key", "apikey", "secret key",
        "access token", "auth token", "private key", "seed phrase", "recovery phrase",
        "recovery code",
    ];
    if CREDENTIALS.iter().any(|k| keyword_value(&lower, k, false)) {
        return finding("secret", "contains a password or credential");
    }
    const CODES: &[&str] = &["pin", "otp", "cvv", "pin code"];
    if CODES.iter().any(|k| keyword_value(&lower, k, true)) {
        return finding("secret", "contains a PIN or one-time code");
    }
    None
}

fn financial(text: &str) -> Option<Finding> {
    for (digits, raw) in digit_runs(text) {
        let n = digits.len();
        // Payment cards: 13-19 digits, Luhn-valid, grouped or contiguous.
        if (13..=19).contains(&n) && luhn(&digits) && !raw.contains('.') {
            return finding("financial", "contains a payment card number");
        }
    }
    let lower = text.to_lowercase();
    const ACCOUNT: &[&str] = &[
        "account number", "account no", "routing number", "iban", "sort code", "swift code",
        "card number", "bank account",
    ];
    for k in ACCOUNT {
        if let Some(pos) = lower.find(k) {
            let window: String = lower[pos..].chars().take(k.len() + 40).collect();
            if window.chars().filter(|c| c.is_ascii_digit()).count() >= 6 {
                return finding("financial", "contains a bank or account number");
            }
        }
    }
    // IBAN-shaped token: 2 letters, 2 digits, 11-30 alphanumerics.
    for t in tokens(text) {
        let c: Vec<char> = t.chars().collect();
        if (15..=34).contains(&c.len())
            && c[0].is_ascii_uppercase()
            && c[1].is_ascii_uppercase()
            && c[2].is_ascii_digit()
            && c[3].is_ascii_digit()
            && c.iter().all(|x| x.is_ascii_uppercase() || x.is_ascii_digit())
        {
            return finding("financial", "contains a bank account number (IBAN)");
        }
    }
    None
}

fn government_id(text: &str) -> Option<Finding> {
    for (digits, raw) in digit_runs(text) {
        // US SSN: ddd-dd-dddd.
        let parts: Vec<&str> = raw.split('-').collect();
        if digits.len() == 9
            && parts.len() == 3
            && parts[0].len() == 3
            && parts[1].len() == 2
            && parts[2].len() == 4
        {
            return finding("government_id", "contains a social security number");
        }
        // Aadhaar: 12 digits grouped 4-4-4.
        let groups: Vec<&str> = raw.split(|c| c == ' ' || c == '-').filter(|g| !g.is_empty()).collect();
        if digits.len() == 12 && groups.len() == 3 && groups.iter().all(|g| g.len() == 4) {
            return finding("government_id", "contains a national ID number");
        }
    }
    for t in tokens(text) {
        // Indian PAN: AAAAA9999A.
        let c: Vec<char> = t.chars().collect();
        if c.len() == 10
            && c[..5].iter().all(|x| x.is_ascii_uppercase())
            && c[5..9].iter().all(|x| x.is_ascii_digit())
            && c[9].is_ascii_uppercase()
        {
            return finding("government_id", "contains a tax ID number");
        }
    }
    let lower = text.to_lowercase();
    for k in ["passport number", "passport no", "license number", "licence number", "ssn"] {
        if let Some(pos) = lower.find(k) {
            let window: String = lower[pos..].chars().take(k.len() + 30).collect();
            if window.chars().filter(|c| c.is_ascii_digit()).count() >= 5 {
                return finding("government_id", "contains an identity document number");
            }
        }
    }
    None
}

fn health(text: &str) -> Option<Finding> {
    const TERMS: &[&str] = &[
        "diagnosed", "diagnosis", "prescription", "prescribed", "medication", "medications",
        "therapy", "therapist", "psychiatrist", "diabetes", "diabetic", "depression",
        "depressed", "anxiety", "adhd", "hiv", "cancer", "chemotherapy", "surgery", "allergic",
        "allergy", "allergies", "symptoms", "pregnant", "pregnancy", "insulin",
        "antidepressant", "antidepressants", "migraine", "migraines", "asthma", "disorder",
        "blood pressure", "blood sugar", "mental health", "sick leave",
    ];
    const SELF: &[&str] = &["i", "i'm", "im", "my", "me", "myself", "i've", "ive"];
    // Only personal health facts: a health term in a first-person sentence.
    for sentence in text.split(|c| matches!(c, '.' | '!' | '?' | '\n')) {
        let lower = sentence.to_lowercase();
        let ws = words(sentence);
        let is_self = ws.iter().any(|w| SELF.contains(&w.as_str())) || lower.contains("i'm ");
        if !is_self {
            continue;
        }
        let hit = TERMS.iter().any(|t| {
            if t.contains(' ') {
                lower.contains(t)
            } else {
                ws.iter().any(|w| w == t)
            }
        });
        if hit {
            return finding("health", "contains personal health information");
        }
    }
    None
}

fn contact(text: &str) -> Option<Finding> {
    for t in tokens(text) {
        if let Some((local, domain)) = t.split_once('@') {
            let tld = domain.rsplit('.').next().unwrap_or("");
            if !local.is_empty()
                && domain.contains('.')
                && tld.len() >= 2
                && tld.chars().all(|c| c.is_ascii_alphabetic())
            {
                return finding("contact", "contains an email address");
            }
        }
    }
    for (digits, raw) in digit_runs(text) {
        let n = digits.len();
        let formatted = raw.starts_with('+')
            || raw.contains('(')
            || raw.matches(|c| c == '-' || c == ' ').count() >= 2;
        // Phone: 10-13 digits, either internationally/visibly formatted or a
        // bare 10-digit mobile number. Dates and decimals do not qualify.
        if (10..=13).contains(&n) && !raw.contains('.') && (formatted || n == 10) {
            return finding("contact", "contains a phone number");
        }
    }
    let lower = text.to_lowercase();
    for k in ["my address is", "i live at", "home address", "my apartment is at"] {
        if lower.contains(k) {
            return finding("contact", "contains a home address");
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::scan;

    fn cat(s: &str) -> Option<&'static str> {
        scan(s).map(|f| f.category)
    }

    #[test]
    fn secrets_stay_local() {
        assert_eq!(cat("my wifi password is hunter2"), Some("secret"));
        assert_eq!(cat("OPENAI key sk-proj-A1b2C3d4E5f6G7h8I9"), Some("secret"));
        assert_eq!(cat("aws AKIAIOSFODNN7EXAMPLE"), Some("secret"));
        assert_eq!(cat("token ghp_16C7e42F292c6912E7710c838347Ae178B4a"), Some("secret"));
        assert_eq!(cat("the bank pin is 4821"), Some("secret"));
        assert_eq!(cat("-----BEGIN RSA PRIVATE KEY-----\nMIIE..."), Some("secret"));
    }

    #[test]
    fn identifiers_and_money_stay_local() {
        assert_eq!(cat("card 4111 1111 1111 1111 exp 09/28"), Some("financial"));
        assert_eq!(cat("my account number is 004512349876"), Some("financial"));
        assert_eq!(cat("IBAN DE89370400440532013000"), Some("financial"));
        assert_eq!(cat("SSN 123-45-6789"), Some("government_id"));
        assert_eq!(cat("aadhaar 1234 5678 9012"), Some("government_id"));
        assert_eq!(cat("PAN ABCDE1234F"), Some("government_id"));
    }

    #[test]
    fn personal_health_and_contact_stay_local() {
        assert_eq!(cat("I'm allergic to peanuts"), Some("health"));
        assert_eq!(cat("I was diagnosed with asthma last year"), Some("health"));
        assert_eq!(cat("email me at yash@example.com"), Some("contact"));
        assert_eq!(cat("call +91 98765 43210 tomorrow"), Some("contact"));
        assert_eq!(cat("my address is 12 MG Road, Pune"), Some("contact"));
    }

    #[test]
    fn ordinary_knowledge_syncs() {
        for s in [
            "reset my password tomorrow",
            "the password is required for login",
            "Cancer detection with CNNs reaches 0.94 AUC on the benchmark",
            "CoreSum TCS is stuck at 0.585 while the target is 0.81",
            "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed the parser",
            "Meeting at 10:30 on 2026-10-02 in room 204",
            "Inference takes 15 to 47 seconds per video",
            "see https://example.com/docs/AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
            "The patient cohort in the study showed fewer symptoms",
            "Practicing piano scales for 20 minutes every day",
            "version 2026.10.02 shipped",
            "build id abc1234567890 passed",
        ] {
            assert_eq!(scan(s), None, "should sync: {s}");
        }
    }
}
