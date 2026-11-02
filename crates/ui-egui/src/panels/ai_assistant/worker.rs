//! The AI Assistant's worker thread: one FIFO queue of questions, a client-side rate limiter
//! for the provider's per-minute budgets, and the HTTP call to the chat-completions endpoint.
//!
//! Rate limiting has three layers, so the per-minute limits are managed, not just retried:
//! 1. a local sliding one-minute window over (requests, estimated tokens);
//! 2. the provider's `x-ratelimit-remaining-*` / `x-ratelimit-reset-*` headers from every
//!    response, which refine the window with the account's live state;
//! 3. on HTTP 429, the `retry-after` header, waited out (cancel-aware), up to three retries.
//!
//! The worker never touches the editing session: it returns a *plan* of engine commands; the
//! panel applies them on the UI thread as one undoable `engine.batch` after the user confirms.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::config::AiConfig;

pub struct Ask {
    pub brief: String,
    pub references: Vec<String>,
    pub context: Value,
    pub history: Vec<(String, String)>,
}

pub enum WorkerMsg {
    Ask(Box<Ask>),
    Cancel,
    Limits { rpm: u32, tpm: u32 },
}

pub enum WorkerEvent {
    Status(String),
    Plan { message: String, commands: Vec<Value> },
    Answer(String),
    Error(String),
    Usage { requests_left_day: Option<u64>, tokens_left_minute: Option<u64> },
}

const SYSTEM: &str = "You are the AI Assistant inside the EffectCraft motion graphics editor. \
The user edits the project by hand; you propose concrete edits as EffectCraft engine commands. \
Reply with exactly one JSON object and no other text: \
{\"message\": string, \"commands\": [{\"command\": string, \"params\": object}]}. \
Rules: use only command ids from the provided catalog; params must match each command's accepted \
keys; keep a plan to at most 12 commands; times are in seconds; never delete project items, \
never render or write files, never change settings; when a step needs an earlier step's result \
(a new layer id, an item id), refer to it as \"$N\" or \"$N.key\" in later params. The commands \
run as one undoable batch. If the request is a question or needs no edit, return an empty \
commands array and put the answer in message. The user's style notes override your defaults.";

struct Reply {
    status: u16,
    retry_after: Option<Duration>,
    tokens_left: Option<u64>,
    tokens_reset: Option<Duration>,
    requests_left_day: Option<u64>,
    body: String,
}

struct Minute {
    stamps: std::collections::VecDeque<(Instant, u64)>,
    hdr_tokens_left: Option<u64>,
    hdr_tokens_reset: Option<Duration>,
    hdr_reqs_left_day: Option<u64>,
}

impl Minute {
    fn new() -> Self {
        Minute { stamps: std::collections::VecDeque::new(), hdr_tokens_left: None, hdr_tokens_reset: None, hdr_reqs_left_day: None }
    }
    fn purge(&mut self) {
        let cut = Instant::now().checked_sub(Duration::from_secs(60));
        while let Some((t, _)) = self.stamps.front() {
            if Some(*t) <= cut { self.stamps.pop_front(); } else { break; }
        }
    }
    fn wait_needed(&mut self, est: u64, rpm: u32, tpm: u32) -> Option<Duration> {
        self.purge();
        if self.hdr_reqs_left_day == Some(0) {
            return Some(Duration::from_secs(90));
        }
        let used: u64 = self.stamps.iter().map(|(_, t)| t).sum();
        let over_local = self.stamps.len() >= rpm as usize || used.saturating_add(est) > tpm as u64;
        let over_hdr = self.hdr_tokens_left.is_some_and(|l| l < est);
        if !over_local && !over_hdr {
            return None;
        }
        let local_wait = self
            .stamps
            .front()
            .and_then(|(t, _)| t.checked_add(Duration::from_secs(60)).and_then(|end| end.checked_duration_since(Instant::now())))
            .unwrap_or(Duration::from_secs(1));
        let hdr_wait = if over_hdr { self.hdr_tokens_reset.unwrap_or(Duration::from_secs(10)) } else { Duration::ZERO };
        Some(local_wait.max(hdr_wait).min(Duration::from_secs(90)))
    }
    fn note(&mut self, r: &Reply) {
        self.hdr_tokens_left = r.tokens_left;
        self.hdr_tokens_reset = r.tokens_reset;
        self.hdr_reqs_left_day = r.requests_left_day;
    }
}

fn parse_reset(s: &str) -> Option<Duration> {
    if let Some(ms) = s.strip_suffix("ms") {
        return ms.parse::<u64>().ok().map(Duration::from_millis);
    }
    let mut secs = 0.0f64;
    let mut cur = String::new();
    let mut found = false;
    for ch in s.chars() {
        match ch {
            '0'..='9' | '.' => cur.push(ch),
            'h' => { secs += cur.parse::<f64>().ok()? * 3600.0; cur.clear(); found = true; }
            'm' => { secs += cur.parse::<f64>().ok()? * 60.0; cur.clear(); found = true; }
            's' => { secs += cur.parse::<f64>().ok()?; cur.clear(); found = true; }
            _ => return None,
        }
    }
    if !cur.is_empty() {
        secs += cur.parse::<f64>().ok()?;
    }
    if found { Some(Duration::from_secs_f64(secs.max(0.0))) } else { None }
}

fn header_u64(h: &ureq::http::HeaderMap, name: &str) -> Option<u64> {
    h.get(name).and_then(|v| v.to_str().ok()).and_then(|s| s.parse::<u64>().ok())
}

fn sleep_cancelable(cancel: &AtomicBool, d: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < d {
        if cancel.load(Ordering::Relaxed) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    cancel.load(Ordering::Relaxed)
}

fn estimate_tokens(system: &str, user: &str) -> u64 {
    (system.len() as u64 + user.len() as u64) / 4 + 2_100
}

fn build_user_message(ask: &Ask, cfg: &AiConfig) -> String {
    let mut msg = String::new();
    if !cfg.style_notes.is_empty() {
        msg.push_str("Style notes (from the user): ");
        msg.push_str(&cfg.style_notes);
        msg.push('\n');
    }
    if !ask.references.is_empty() {
        msg.push_str(&format!("Reference material the user added: {}\n", ask.references.join(", ")));
    }
    for (role, text) in ask.history.iter().rev().take(6).rev() {
        msg.push_str(&format!("{role}: {text}\n"));
    }
    msg.push_str(&format!("Context: {}\n", ask.context));
    msg.push_str(&format!("Request: {}", ask.brief));
    msg
}

fn parse_plan(content: &str) -> Result<(String, Vec<Value>), String> {
    let v: Value = serde_json::from_str(content).map_err(|e| format!("the model did not return JSON: {e}"))?;
    let message = v.get("message").and_then(Value::as_str).unwrap_or("").to_string();
    let commands = v
        .get("commands")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|c| c.get("command").and_then(Value::as_str).is_some())
        .take(12)
        .collect();
    Ok((message, commands))
}

fn http_post(agent: &ureq::Agent, cfg: &AiConfig, body: &Value) -> Result<Reply, String> {
    let auth = format!("Bearer {}", cfg.api_key);
    let resp = agent
        .post(&cfg.endpoint)
        .header("Authorization", auth.as_str())
        .header("Content-Type", "application/json")
        .send(body.to_string().as_bytes())
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status().as_u16();
    let h = resp.headers();
    let retry_after = h.get("retry-after").and_then(|v| v.to_str().ok()).and_then(parse_reset);
    let tokens_left = header_u64(h, "x-ratelimit-remaining-tokens");
    let tokens_reset = h.get("x-ratelimit-reset-tokens").and_then(|v| v.to_str().ok()).and_then(parse_reset);
    let requests_left_day = header_u64(h, "x-ratelimit-remaining-requests");
    let body_text = resp.into_body().read_to_string().map_err(|e| format!("cannot read the reply: {e}"))?;
    Ok(Reply { status, retry_after, tokens_left, tokens_reset, requests_left_day, body: body_text })
}

fn gate(minute: &mut Minute, est: u64, cfg: &AiConfig, cancel: &AtomicBool, tx: &Sender<WorkerEvent>) -> Result<(), String> {
    if let Some(wait) = minute.wait_needed(est, cfg.requests_per_minute, cfg.tokens_per_minute) {
        let _ = tx.send(WorkerEvent::Status(format!("rate limit: waiting {}s", wait.as_secs())));
        if sleep_cancelable(cancel, wait) {
            return Err("cancelled".to_string());
        }
        if minute.wait_needed(est, cfg.requests_per_minute, cfg.tokens_per_minute).is_some() {
            return Err("rate limit reached; try again in a minute".to_string());
        }
    }
    Ok(())
}

fn request_with_retries(agent: &ureq::Agent, cfg: &AiConfig, minute: &mut Minute, body: &Value, cancel: &AtomicBool, tx: &Sender<WorkerEvent>) -> Result<Reply, String> {
    let mut attempt = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let reply = http_post(agent, cfg, body)?;
        minute.note(&reply);
        let _ = tx.send(WorkerEvent::Usage { requests_left_day: reply.requests_left_day, tokens_left_minute: reply.tokens_left });
        if reply.status == 429 && attempt < 3 {
            attempt += 1;
            let wait = reply.retry_after.unwrap_or(Duration::from_secs(5));
            let _ = tx.send(WorkerEvent::Status(format!("provider rate limit: retrying in {}s", wait.as_secs())));
            if sleep_cancelable(cancel, wait) {
                return Err("cancelled".to_string());
            }
            continue;
        }
        return Ok(reply);
    }
}

fn ask_once(agent: &ureq::Agent, cfg: &AiConfig, minute: &mut Minute, ask: &Ask, cancel: &AtomicBool, tx: &Sender<WorkerEvent>) -> Result<WorkerEvent, String> {
    let user = build_user_message(ask, cfg);
    let est = estimate_tokens(SYSTEM, &user);
    gate(minute, est, cfg, cancel, tx)?;
    let body = json!({
        "model": cfg.model,
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    });
    let _ = tx.send(WorkerEvent::Status("asking the model".to_string()));
    let reply = request_with_retries(agent, cfg, minute, &body, cancel, tx)?;
    if reply.status != 200 {
        let msg: String = reply.body.chars().take(300).collect();
        return Err(format!("the model API returned HTTP {}: {msg}", reply.status));
    }
    minute.stamps.push_back((Instant::now(), est));
    let v: Value = serde_json::from_str(&reply.body).map_err(|e| format!("bad API reply: {e}"))?;
    let content = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| "the API reply had no message".to_string())?;
    let (message, commands) = parse_plan(content)?;
    if commands.is_empty() {
        return Ok(WorkerEvent::Answer(message));
    }
    Ok(WorkerEvent::Plan { message, commands })
}

pub fn spawn(cfg: AiConfig) -> (Sender<WorkerMsg>, Receiver<WorkerEvent>) {
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<WorkerMsg>();
    let (ev_tx, ev_rx) = std::sync::mpsc::channel::<WorkerEvent>();
    std::thread::spawn(move || {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(180)))
            .http_status_as_error(false)
            .build()
            .into();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cfg = cfg;
        let mut minute = Minute::new();
        while let Ok(msg) = msg_rx.recv() {
            match msg {
                WorkerMsg::Cancel => {
                    cancel.store(true, Ordering::Relaxed);
                }
                WorkerMsg::Limits { rpm, tpm } => {
                    cfg.requests_per_minute = rpm;
                    cfg.tokens_per_minute = tpm;
                }
                WorkerMsg::Ask(ask) => {
                    if cfg.api_key.is_empty() {
                        let _ = ev_tx.send(WorkerEvent::Error("no API key: add your Groq key in the panel's settings".to_string()));
                        continue;
                    }
                    cancel.store(false, Ordering::Relaxed);
                    match ask_once(&agent, &cfg, &mut minute, &ask, &cancel, &ev_tx) {
                        Ok(ev) => {
                            let _ = ev_tx.send(ev);
                        }
                        Err(e) => {
                            let _ = ev_tx.send(WorkerEvent::Error(e));
                        }
                    }
                }
            }
        }
    });
    (msg_tx, ev_rx)
}

#[cfg(test)]
mod tests {
    //! Offline integration tests: the worker talks to a local mock server, never a real
    //! provider. They prove the request shape (auth header, model, JSON mode), plan parsing,
    //! usage-header tracking and the 429/retry-after path without spending a single real call.
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    struct Mock {
        addr: std::net::SocketAddr,
        requests: Receiver<(String, String)>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl Mock {
        /// Serves `responses`, one per request, in order.
        fn start(responses: Vec<String>) -> Mock {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                for response in responses {
                    let (stream, _) = listener.accept().unwrap();
                    let mut reader = BufReader::new(stream);
                    let mut auth = String::new();
                    let mut content_length = 0usize;
                    let mut line = String::new();
                    // Request line, then headers.
                    reader.read_line(&mut line).unwrap();
                    loop {
                        let mut h = String::new();
                        reader.read_line(&mut h).unwrap();
                        let h = h.trim();
                        if h.is_empty() {
                            break;
                        }
                        if let Some((name, value)) = h.split_once(':') {
                            if name.eq_ignore_ascii_case("authorization") {
                                auth = value.trim().to_string();
                            }
                            if name.eq_ignore_ascii_case("content-length") {
                                content_length = value.trim().parse().unwrap_or(0);
                            }
                        }
                    }
                    let mut body = vec![0u8; content_length];
                    reader.read_exact(&mut body).unwrap();
                    let _ = tx.send((auth, String::from_utf8_lossy(&body).to_string()));
                    let mut stream = reader.into_inner();
                    stream.write_all(response.as_bytes()).unwrap();
                    stream.flush().unwrap();
                }
            });
            Mock { addr, requests: rx, handle: Some(handle) }
        }

        fn cfg(&self) -> AiConfig {
            AiConfig {
                api_key: "test-key".to_string(),
                endpoint: format!("http://{}/v1/chat/completions", self.addr),
                model: "test-model".to_string(),
                requests_per_minute: 600,
                tokens_per_minute: 10_000_000,
                ..AiConfig::default()
            }
        }
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn http(status: &str, headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn api_reply(plan_json: &str) -> String {
        let content = plan_json.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        http(
            "200 OK",
            "x-ratelimit-remaining-requests: 700\r\nx-ratelimit-remaining-tokens: 12345\r\n",
            &format!("{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{content}\"}}}}]}}"),
        )
    }

    fn ask(brief: &str) -> Box<Ask> {
        Box::new(Ask {
            brief: brief.to_string(),
            references: vec!["user-ref: bold kinetic type".to_string()],
            context: json!({"compositions": []}),
            history: vec![],
        })
    }

    /// Collect events until a terminal one (Plan/Answer/Error), or time out.
    fn collect(rx: &Receiver<WorkerEvent>) -> (Vec<WorkerEvent>, WorkerEvent) {
        let mut all = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            // The worker legitimately goes quiet while sleeping out a 429 retry-after,
            // so wait out the whole remaining budget for each event.
            let remaining = deadline.saturating_duration_since(Instant::now());
            let ev = rx.recv_timeout(remaining).expect("worker produced no terminal event in time");
            if matches!(&ev, WorkerEvent::Plan { .. } | WorkerEvent::Answer(_) | WorkerEvent::Error(_)) {
                return (all, ev);
            }
            all.push(ev);
        }
    }

    #[test]
    fn plan_request_shape_and_parse() {
        let mock = Mock::start(vec![api_reply(
            "{\"message\":\"adding a title\",\"commands\":[{\"command\":\"layer.addText\",\"params\":{\"name\":\"Title\"}}]}",
        )]);
        let (tx, rx) = spawn(mock.cfg());
        tx.send(WorkerMsg::Ask(ask("add a title"))).unwrap();
        let (prior, terminal) = collect(&rx);
        match terminal {
            WorkerEvent::Plan { message, commands } => {
                assert_eq!(message, "adding a title");
                assert_eq!(commands.len(), 1);
                assert_eq!(commands[0]["command"], "layer.addText");
            }
            _ => panic!("expected a plan"),
        }
        // The provider's usage headers are surfaced before the plan.
        assert!(prior.iter().any(|e| matches!(
            e,
            WorkerEvent::Usage { requests_left_day: Some(700), tokens_left_minute: Some(12345) }
        )));
        // The request itself: auth header, model id, JSON mode.
        let (auth, body) = mock.requests.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(auth, "Bearer test-key");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["response_format"]["type"], "json_object");
        let user_msg = body["messages"][1]["content"].as_str().unwrap();
        assert!(user_msg.contains("add a title"));
        assert!(user_msg.contains("user-ref: bold kinetic type"));
    }

    #[test]
    fn retries_once_on_429_then_answers() {
        let too_many = http("429 Too Many Requests", "retry-after: 0\r\n", "{\"error\":\"rate limited\"}");
        let mock = Mock::start(vec![too_many, api_reply("{\"message\":\"just a question\",\"commands\":[]}")]);
        let (tx, rx) = spawn(mock.cfg());
        tx.send(WorkerMsg::Ask(ask("what comps exist?"))).unwrap();
        let (prior, terminal) = collect(&rx);
        match terminal {
            WorkerEvent::Answer(m) => assert_eq!(m, "just a question"),
            _ => panic!("expected an answer"),
        }
        assert!(prior.iter().any(|e| matches!(e, WorkerEvent::Status(s) if s.contains("provider rate limit"))));
        // Both attempts hit the mock.
        mock.requests.recv_timeout(Duration::from_secs(5)).unwrap();
        mock.requests.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn non_json_model_reply_is_an_error_not_a_crash() {
        let mock = Mock::start(vec![api_reply("sorry, I cannot help with that")]);
        let (tx, rx) = spawn(mock.cfg());
        tx.send(WorkerMsg::Ask(ask("add a title"))).unwrap();
        let (_, terminal) = collect(&rx);
        match terminal {
            WorkerEvent::Error(e) => assert!(e.contains("did not return JSON"), "{e}"),
            _ => panic!("expected an error"),
        }
    }

    #[test]
    fn missing_key_is_a_local_error_no_http_call() {
        let mock = Mock::start(vec![]);
        let mut cfg = mock.cfg();
        cfg.api_key = String::new();
        let (tx, rx) = spawn(cfg);
        tx.send(WorkerMsg::Ask(ask("add a title"))).unwrap();
        let (_, terminal) = collect(&rx);
        match terminal {
            WorkerEvent::Error(e) => assert!(e.contains("no API key"), "{e}"),
            _ => panic!("expected an error"),
        }
        assert!(mock.requests.recv_timeout(Duration::from_millis(500)).is_err());
    }
}
