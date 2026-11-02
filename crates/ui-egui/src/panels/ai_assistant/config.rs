//! The AI Assistant's per-user settings: `ai_assistant.toml` in EffectCraft's config directory
//! (the same folder the session's settings live in). Holds the Groq API key, the model and the
//! rate limits. The file is written owner-readable only; the key is never logged and never
//! leaves the machine except in the Authorization header of a request to the configured
//! endpoint. `GROQ_API_KEY` in the environment overrides the file's `api_key`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    /// Groq API key (gsk_...). Empty until the user saves one.
    pub api_key: String,
    /// Chat-completions endpoint.
    pub endpoint: String,
    /// Model id, e.g. `llama-3.3-70b-versatile`.
    pub model: String,
    /// Client-side requests-per-minute budget (the provider's headers refine it live).
    pub requests_per_minute: u32,
    /// Client-side tokens-per-minute budget.
    pub tokens_per_minute: u32,
    /// Cap on the serialized project context sent with one question.
    pub max_context_chars: usize,
    /// The user's standing style notes, sent with every question.
    pub style_notes: String,
}

impl Default for AiConfig {
    fn default() -> Self {
        AiConfig {
            api_key: String::new(),
            endpoint: "https://api.groq.com/openai/v1/chat/completions".to_string(),
            model: "llama-3.3-70b-versatile".to_string(),
            requests_per_minute: 28,
            tokens_per_minute: 6_000,
            max_context_chars: 12_000,
            style_notes: String::new(),
        }
    }
}

pub fn path(config_dir: Option<&Path>) -> Option<PathBuf> {
    config_dir.map(|d| d.join("ai_assistant.toml"))
}

pub fn load(config_dir: Option<&Path>) -> AiConfig {
    let mut cfg = path(config_dir)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str::<AiConfig>(&s).ok())
        .unwrap_or_default();
    if let Ok(k) = std::env::var("GROQ_API_KEY")
        && !k.is_empty()
    {
        cfg.api_key = k;
    }
    cfg
}

pub fn save(config_dir: Option<&Path>, cfg: &AiConfig) -> Result<(), String> {
    let Some(p) = path(config_dir) else {
        return Err("no config directory".to_string());
    };
    let s = toml::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(&p, s).map_err(|e| format!("cannot write {}: {e}", p.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_sets_owner_only_permissions_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("ai-cfg-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = AiConfig {
            api_key: "gsk_secret".to_string(),
            model: "test-model".to_string(),
            requests_per_minute: 10,
            tokens_per_minute: 1000,
            style_notes: "punchy".to_string(),
            ..AiConfig::default()
        };
        save(Some(&dir), &cfg).unwrap();
        let p = dir.join("ai_assistant.toml");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // Load reads the file back; the env override must not leak a real key into the test.
        let prev = std::env::var("GROQ_API_KEY").ok();
        unsafe { std::env::remove_var("GROQ_API_KEY") };
        let loaded = load(Some(&dir));
        if let Some(v) = prev {
            unsafe { std::env::set_var("GROQ_API_KEY", v) };
        }
        assert_eq!(loaded.api_key, "gsk_secret");
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.requests_per_minute, 10);
        assert_eq!(loaded.tokens_per_minute, 1000);
        assert_eq!(loaded.style_notes, "punchy");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_without_dir_uses_defaults() {
        let cfg = load(None);
        assert!(cfg.endpoint.contains("groq"));
    }
}
