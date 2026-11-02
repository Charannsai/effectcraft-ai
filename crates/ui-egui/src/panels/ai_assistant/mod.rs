//! The AI Assistant panel: a chat-style side panel whose answers are concrete EffectCraft
//! edit plans. The user describes an edit (with their own style notes and reference material);
//! a worker thread asks the configured LLM (Groq by default) for a plan of engine commands,
//! the panel shows the plan, and Apply runs it as ONE undoable `engine.batch` on the session,
//! then renders a preview frame. Every edit the assistant makes is inspectable here and
//! undoable with Edit ▸ Undo like any manual edit. The worker manages the provider's
//! per-minute rate limits (see `worker`).
//!
//! Automation ids: `ai.scroll`, `ai.brief`, `ai.send`, `ai.cancel`, `ai.plan.apply`,
//! `ai.plan.discard`, `ai.undo`, `ai.reference.field`, `ai.reference.add`,
//! `ai.reference.remove.<n>`, `ai.settings.toggle`, `ai.settings.key`, `ai.settings.model`,
//! `ai.settings.rpm`, `ai.settings.tpm`, `ai.settings.notes`, `ai.settings.save`.
//!
//! Native only: the panel opens in the browser build but the worker needs threads and TLS.

#[cfg(not(target_arch = "wasm32"))]
mod config;
#[cfg(not(target_arch = "wasm32"))]
mod worker;

use egui::{Align2, Rect, pos2, vec2};
#[cfg(not(target_arch = "wasm32"))]
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
use crate::automation::Registry;
use crate::i18n::tr;
use crate::theme::Tokens;
#[cfg(not(target_arch = "wasm32"))]
use crate::widgets;
use crate::EffectcraftApp;

#[cfg(not(target_arch = "wasm32"))]
use config::AiConfig;

#[cfg(not(target_arch = "wasm32"))]
pub struct PlanState {
    message: String,
    commands: Vec<Value>,
}

#[derive(Default)]
pub struct AiAssistantState {
    #[cfg(not(target_arch = "wasm32"))]
    cfg: AiConfig,
    #[cfg(not(target_arch = "wasm32"))]
    cfg_loaded: bool,
    #[cfg(not(target_arch = "wasm32"))]
    cfg_dir: Option<std::path::PathBuf>,
    brief: String,
    references: Vec<String>,
    ref_input: String,
    chat: Vec<(String, String)>,
    #[cfg(not(target_arch = "wasm32"))]
    plan: Option<PlanState>,
    status: String,
    busy: bool,
    usage: String,
    preview: Option<egui::TextureHandle>,
    settings_open: bool,
    edit_key: String,
    edit_model: String,
    edit_rpm: String,
    edit_tpm: String,
    edit_notes: String,
    #[cfg(not(target_arch = "wasm32"))]
    to_worker: Option<std::sync::mpsc::Sender<worker::WorkerMsg>>,
    #[cfg(not(target_arch = "wasm32"))]
    from_worker: Option<std::sync::mpsc::Receiver<worker::WorkerEvent>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AiAssistantState {
    fn ensure_loaded(&mut self, dir: Option<std::path::PathBuf>) {
        if self.cfg_loaded {
            return;
        }
        self.cfg = config::load(dir.as_deref());
        self.edit_key = self.cfg.api_key.clone();
        self.edit_model = self.cfg.model.clone();
        self.edit_rpm = self.cfg.requests_per_minute.to_string();
        self.edit_tpm = self.cfg.tokens_per_minute.to_string();
        self.edit_notes = self.cfg.style_notes.clone();
        self.cfg_dir = dir;
        self.cfg_loaded = true;
    }

    fn ensure_worker(&mut self) {
        if self.to_worker.is_none() {
            let (tx, rx) = worker::spawn(self.cfg.clone());
            self.to_worker = Some(tx);
            self.from_worker = Some(rx);
        }
    }

    fn poll(&mut self) {
        let events: Vec<worker::WorkerEvent> = self.from_worker.as_ref().map(|rx| rx.try_iter().collect()).unwrap_or_default();
        for ev in events {
            match ev {
                worker::WorkerEvent::Status(s) => self.status = s,
                worker::WorkerEvent::Usage { requests_left_day, tokens_left_minute } => {
                    let d = requests_left_day.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
                    let t = tokens_left_minute.map(|n| n.to_string()).unwrap_or_else(|| "?".to_string());
                    self.usage = format!("{d} requests today, {t} tokens this minute left");
                }
                worker::WorkerEvent::Plan { message, commands } => {
                    self.busy = false;
                    self.status = tr("plan ready: review and Apply").to_string();
                    self.plan = Some(PlanState { message, commands });
                }
                worker::WorkerEvent::Answer(m) => {
                    self.busy = false;
                    self.status.clear();
                    self.chat.push(("assistant".to_string(), m));
                }
                worker::WorkerEvent::Error(e) => {
                    self.busy = false;
                    self.status = e;
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum Act {
    None,
    Send,
    Cancel,
    Apply,
    Discard,
    Undo,
    AddRef,
    RemoveRef(usize),
    SaveCfg,
}

#[cfg(not(target_arch = "wasm32"))]
fn reg(auto: &mut Registry, id: &str, r: &egui::Response, label: &str) {
    auto.add(id, r.rect, label);
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_context(s: &mut effectcraft_engine::Session, max_chars: usize) -> Value {
    let summary = s.execute("project.summary", serde_json::json!({})).unwrap_or_else(|_| serde_json::json!({}));
    let comp = s.execute("comp.info", serde_json::json!({})).unwrap_or_else(|_| serde_json::json!({}));
    let catalog = s.execute("command.list", serde_json::json!({"enabledOnly": true})).unwrap_or_else(|_| serde_json::json!([]));
    let mut ids: Vec<Value> = catalog
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| {
                    let id = c.get("id").and_then(Value::as_str)?;
                    let label = c.get("label").and_then(Value::as_str).unwrap_or(id);
                    Some(serde_json::json!({"id": id, "label": label}))
                })
                .collect()
        })
        .unwrap_or_default();
    loop {
        let ctx = serde_json::json!({"project": summary, "comp": comp, "commands": ids});
        if serde_json::to_string(&ctx).map(|s| s.len()).unwrap_or(0) <= max_chars || ids.is_empty() {
            return ctx;
        }
        let keep = ids.len().saturating_sub(ids.len() / 4 + 1);
        ids.truncate(keep);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn send_brief(app: &mut EffectcraftApp) {
    let brief = app.ai_assistant.brief.trim().to_string();
    if brief.is_empty() {
        return;
    }
    let max = app.ai_assistant.cfg.max_context_chars;
    let context = collect_context(&mut app.session, max);
    let st = &mut app.ai_assistant;
    st.ensure_worker();
    st.chat.push(("you".to_string(), brief.clone()));
    let history: Vec<(String, String)> = st.chat.iter().rev().take(6).rev().cloned().collect();
    st.brief.clear();
    st.busy = true;
    st.status = tr("queued").to_string();
    let ask = worker::Ask { brief, references: st.references.clone(), context, history };
    if let Some(tx) = &st.to_worker {
        let _ = tx.send(worker::WorkerMsg::Ask(Box::new(ask)));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn render_preview(app: &mut EffectcraftApp, ctx: &egui::Context) {
    let Ok(cid) = app.session.resolve_comp(None) else {
        return;
    };
    let t = app.session.time();
    let Ok((w, h, rgba)) = app.session.render_rgba8_alpha(cid, t, 480, false) else {
        return;
    };
    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
    app.ai_assistant.preview = Some(ctx.load_texture("ai-assistant-preview", img, egui::TextureOptions::LINEAR));
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_plan(app: &mut EffectcraftApp, ctx: &egui::Context) {
    let Some(plan) = app.ai_assistant.plan.take() else {
        return;
    };
    let n = plan.commands.len();
    let r = app.session.execute_checked("engine.batch", serde_json::json!({"steps": plan.commands, "label": "AI Assistant edit", "atomic": true}));
    let st = &mut app.ai_assistant;
    match r {
        Ok(_) => {
            st.chat.push(("assistant".to_string(), format!("Applied {n} command(s) as one undo step.")));
            st.status.clear();
        }
        Err(e) => {
            st.chat.push(("assistant".to_string(), format!("Could not apply the plan: {e}")));
            st.status = tr("apply failed").to_string();
        }
    }
    render_preview(app, ctx);
}

#[cfg(not(target_arch = "wasm32"))]
fn save_cfg(st: &mut AiAssistantState) {
    st.cfg.api_key = st.edit_key.trim().to_string();
    if !st.edit_model.trim().is_empty() {
        st.cfg.model = st.edit_model.trim().to_string();
    }
    if let Ok(rpm) = st.edit_rpm.trim().parse::<u32>() {
        st.cfg.requests_per_minute = rpm.max(1);
    }
    if let Ok(tpm) = st.edit_tpm.trim().parse::<u32>() {
        st.cfg.tokens_per_minute = tpm.max(1_000);
    }
    st.cfg.style_notes = st.edit_notes.clone();
    st.status = match config::save(st.cfg_dir.as_deref(), &st.cfg) {
        Ok(()) => tr("settings saved").to_string(),
        Err(e) => e,
    };
    let (rpm, tpm) = (st.cfg.requests_per_minute, st.cfg.tokens_per_minute);
    st.ensure_worker();
    if let Some(tx) = &st.to_worker {
        let _ = tx.send(worker::WorkerMsg::Limits { rpm, tpm });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_action(app: &mut EffectcraftApp, ctx: &egui::Context, act: Act) {
    match act {
        Act::None => {}
        Act::Send => send_brief(app),
        Act::Cancel => {
            let st = &mut app.ai_assistant;
            if let Some(tx) = &st.to_worker {
                let _ = tx.send(worker::WorkerMsg::Cancel);
            }
            st.busy = false;
            st.status = tr("cancelled").to_string();
        }
        Act::Apply => apply_plan(app, ctx),
        Act::Discard => {
            app.ai_assistant.plan = None;
        }
        Act::Undo => {
            let r = app.session.execute("edit.undo", serde_json::json!({}));
            app.ai_assistant.status = match r {
                Ok(_) => tr("undone").to_string(),
                Err(e) => format!("undo failed: {e}"),
            };
            render_preview(app, ctx);
        }
        Act::AddRef => {
            let st = &mut app.ai_assistant;
            let r = st.ref_input.trim().to_string();
            if !r.is_empty() {
                st.references.push(r);
                st.ref_input.clear();
            }
        }
        Act::RemoveRef(i) => {
            let st = &mut app.ai_assistant;
            if i < st.references.len() {
                st.references.remove(i);
            }
        }
        Act::SaveCfg => save_cfg(&mut app.ai_assistant),
    }
}

pub fn show(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect) {
    #[cfg(target_arch = "wasm32")]
    {
        let p = ui.painter().with_clip_rect(rect);
        p.text(rect.min + vec2(12.0, 14.0), Align2::LEFT_TOP, tr("The AI Assistant runs in the desktop app."), Tokens::ui(13.0), app.tokens.text);
    }
    #[cfg(not(target_arch = "wasm32"))]
    show_native(app, ui, rect);
}

#[cfg(not(target_arch = "wasm32"))]
struct Lay {
    x0: f32,
    w: f32,
    y: f32,
}

#[cfg(not(target_arch = "wasm32"))]
fn field(ui: &mut egui::Ui, auto: &mut Registry, t: &Tokens, id: &str, hint: &str, rect: Rect, text: &mut String) -> egui::Response {
    let r = widgets::text_field(ui, rect, text, hint, t);
    auto.add(id, r.rect, hint);
    r
}

#[cfg(not(target_arch = "wasm32"))]
fn wrapped(ui: &egui::Ui, t: &Tokens, x: f32, y: f32, w: f32, text: &str) -> f32 {
    let g = ui.painter().layout(text.to_string(), Tokens::ui(12.0), t.text, w);
    let h = g.size().y;
    ui.painter().galley(pos2(x, y), g, t.text);
    h
}

#[cfg(not(target_arch = "wasm32"))]
fn sec_chat(app: &mut EffectcraftApp, ui: &mut egui::Ui, lay: &mut Lay) {
    let t = app.tokens;
    let history: Vec<(String, String)> = app.ai_assistant.chat.iter().rev().take(20).rev().cloned().collect();
    for (role, text) in &history {
        let label = if role == "you" { tr("You") } else { tr("Assistant") };
        ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, label, Tokens::ui(11.0), t.text_dim);
        lay.y += 14.0;
        lay.y += wrapped(ui, &t, lay.x0, lay.y, lay.w, text) + 8.0;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sec_plan(app: &mut EffectcraftApp, ui: &mut egui::Ui, lay: &mut Lay) -> Act {
    let t = app.tokens;
    let mut act = Act::None;
    let Some(plan) = &app.ai_assistant.plan else {
        return act;
    };
    lay.y += wrapped(ui, &t, lay.x0, lay.y, lay.w, &plan.message) + 6.0;
    for (i, c) in plan.commands.iter().enumerate() {
        let id = c.get("command").and_then(Value::as_str).unwrap_or("?");
        let params = c.get("params").map(|p| p.to_string()).unwrap_or_default();
        // Full command and params, wrapped to the panel width: review must show everything.
        let line = format!("{}. {} {}", i + 1, id, params);
        let g = ui.painter().layout(line, Tokens::mono(11.0), t.text_dim, lay.w - 8.0);
        let h = g.size().y;
        ui.painter().galley(pos2(lay.x0 + 8.0, lay.y), g, t.text_dim);
        lay.y += h + 4.0;
    }
    lay.y += 6.0;
    let apply = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0, lay.y), vec2(70.0, 22.0)), tr("Apply"), true, &t, egui::Id::unique("ai-apply"));
    reg(&mut app.auto, "ai.plan.apply", &apply, "Apply plan");
    if apply.clicked() {
        act = Act::Apply;
    }
    let discard = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0 + 78.0, lay.y), vec2(70.0, 22.0)), tr("Discard"), false, &t, egui::Id::unique("ai-discard"));
    reg(&mut app.auto, "ai.plan.discard", &discard, "Discard plan");
    if discard.clicked() {
        act = Act::Discard;
    }
    lay.y += 30.0;
    act
}

#[cfg(not(target_arch = "wasm32"))]
fn sec_input(app: &mut EffectcraftApp, ui: &mut egui::Ui, lay: &mut Lay) -> Act {
    let t = app.tokens;
    let mut act = Act::None;
    let busy = app.ai_assistant.busy;
    let resp = {
        let st = &mut app.ai_assistant;
        field(ui, &mut app.auto, &t, "ai.brief", tr("Describe the edit..."), Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w - 74.0, 22.0)), &mut st.brief)
    };
    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        act = Act::Send;
    }
    let (label, primary, id, auto_id, clicked_act) = if busy {
        (tr("Cancel"), false, "ai-cancel", "ai.cancel", Act::Cancel)
    } else {
        (tr("Plan"), true, "ai-send", "ai.send", Act::Send)
    };
    let b = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0 + lay.w - 66.0, lay.y), vec2(66.0, 22.0)), label, primary, &t, egui::Id::unique(id));
    reg(&mut app.auto, auto_id, &b, label);
    if b.clicked() {
        act = clicked_act;
    }
    lay.y += 28.0;
    if !app.ai_assistant.status.is_empty() {
        ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, app.ai_assistant.status.clone(), Tokens::ui(11.0), t.text_dim);
        lay.y += 16.0;
    }
    lay.y += 4.0;
    act
}

#[cfg(not(target_arch = "wasm32"))]
fn sec_refs(app: &mut EffectcraftApp, ui: &mut egui::Ui, lay: &mut Lay) -> Act {
    let t = app.tokens;
    let mut act = Act::None;
    ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, tr("References (click one to remove it)"), Tokens::ui(11.0), t.text_dim);
    lay.y += 15.0;
    // Wrapped with the panel's width: the caveat must be fully readable at the default narrow width.
    let caveat = tr("Paths and notes are sent to the model as text only; files are not read, watched or uploaded.");
    let galley = ui.painter().layout(caveat.to_string(), Tokens::ui(11.0), t.text_dim, lay.w);
    let h = galley.size().y;
    ui.painter().galley(pos2(lay.x0, lay.y), galley, t.text_dim);
    lay.y += h + 4.0;
    let refs = app.ai_assistant.references.clone();
    for (i, r) in refs.iter().enumerate() {
        let rr = Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w, 15.0));
        let resp = ui.interact(rr, egui::Id::unique(("ai-ref", i)), egui::Sense::click());
        let line = format!("- {}", r.chars().take(70).collect::<String>());
        ui.painter().text(pos2(lay.x0 + 8.0, lay.y), Align2::LEFT_TOP, line, Tokens::ui(11.0), t.text);
        reg(&mut app.auto, &format!("ai.reference.remove.{i}"), &resp, "Remove reference");
        if resp.clicked() {
            act = Act::RemoveRef(i);
        }
        lay.y += 16.0;
    }
    let resp = {
        let st = &mut app.ai_assistant;
        field(ui, &mut app.auto, &t, "ai.reference.field", tr("Add a reference path..."), Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w - 74.0, 22.0)), &mut st.ref_input)
    };
    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        act = Act::AddRef;
    }
    let add = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0 + lay.w - 66.0, lay.y), vec2(66.0, 22.0)), tr("Add"), false, &t, egui::Id::unique("ai-addref"));
    reg(&mut app.auto, "ai.reference.add", &add, "Add reference");
    if add.clicked() {
        act = Act::AddRef;
    }
    lay.y += 30.0;
    act
}

#[cfg(not(target_arch = "wasm32"))]
fn sec_settings(app: &mut EffectcraftApp, ui: &mut egui::Ui, lay: &mut Lay) -> Act {
    let t = app.tokens;
    let mut act = Act::None;
    let set = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0, lay.y), vec2(90.0, 22.0)), tr("Settings"), false, &t, egui::Id::unique("ai-settings"));
    reg(&mut app.auto, "ai.settings.toggle", &set, "Toggle settings");
    if set.clicked() {
        app.ai_assistant.settings_open = !app.ai_assistant.settings_open;
    }
    lay.y += 28.0;
    if !app.ai_assistant.settings_open {
        return act;
    }
    let st = &mut app.ai_assistant;
    ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, tr("Groq API key (saved in your private config folder)"), Tokens::ui(11.0), t.text_dim);
    lay.y += 15.0;
    let half = (lay.w - 8.0) / 2.0;
    // Masked by default; revealed only while "Show" is held down, so a saved key never sits
    // on screen during a screenshare or screenshot.
    let show = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0 + lay.w - 44.0, lay.y), vec2(44.0, 22.0)), tr("Show"), false, &t, egui::Id::unique("ai-key-show"));
    reg(&mut app.auto, "ai.settings.keyReveal", &show, "Hold to reveal the API key");
    let reveal = show.is_pointer_button_down_on();
    let key_rect = Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w - 50.0, 22.0));
    let key_resp = widgets::text_field_secret(ui, key_rect, &mut st.edit_key, "gsk_...", &t, reveal);
    app.auto.add("ai.settings.key", key_resp.rect, "gsk_...");
    lay.y += 26.0;
    field(ui, &mut app.auto, &t, "ai.settings.model", tr("model"), Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w, 22.0)), &mut st.edit_model);
    lay.y += 26.0;
    field(ui, &mut app.auto, &t, "ai.settings.rpm", tr("requests per minute"), Rect::from_min_size(pos2(lay.x0, lay.y), vec2(half, 22.0)), &mut st.edit_rpm);
    field(ui, &mut app.auto, &t, "ai.settings.tpm", tr("tokens per minute"), Rect::from_min_size(pos2(lay.x0 + half + 8.0, lay.y), vec2(half, 22.0)), &mut st.edit_tpm);
    lay.y += 26.0;
    field(ui, &mut app.auto, &t, "ai.settings.notes", tr("style notes (sent with every request)"), Rect::from_min_size(pos2(lay.x0, lay.y), vec2(lay.w, 22.0)), &mut st.edit_notes);
    lay.y += 26.0;
    let save = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0, lay.y), vec2(110.0, 22.0)), tr("Save settings"), true, &t, egui::Id::unique("ai-save"));
    reg(&mut app.auto, "ai.settings.save", &save, "Save settings");
    if save.clicked() {
        act = Act::SaveCfg;
    }
    lay.y += 30.0;
    act
}

#[cfg(not(target_arch = "wasm32"))]
fn show_native(app: &mut EffectcraftApp, ui: &mut egui::Ui, rect: Rect) {
    app.ai_assistant.ensure_loaded(effectcraft_host::config_dir());
    app.ai_assistant.poll();
    if app.ai_assistant.busy {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    }
    let t = app.tokens;
    let scroll = widgets::PanelScroll::begin(ui, egui::Id::unique("ai-scroll"), rect);
    let clip = ui.clip_rect();
    ui.set_clip_rect(rect.intersect(clip));
    let mut lay = Lay { x0: rect.min.x + 12.0, w: (rect.width() - 30.0).max(80.0), y: rect.min.y + 12.0 - scroll.offset };

    ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, tr("AI Assistant"), Tokens::ui(15.0), t.text);
    lay.y += 24.0;
    if !app.ai_assistant.usage.is_empty() {
        ui.painter().text(pos2(lay.x0, lay.y), Align2::LEFT_TOP, app.ai_assistant.usage.clone(), Tokens::ui(11.0), t.text_dim);
        lay.y += 16.0;
    }
    lay.y += 4.0;

    sec_chat(app, ui, &mut lay);
    let mut act = sec_plan(app, ui, &mut lay);

    // Undo the last edit (the assistant's plans apply as one undo step).
    let undo = widgets::text_button(ui, Rect::from_min_size(pos2(lay.x0, lay.y), vec2(120.0, 22.0)), tr("Undo last edit"), false, &t, egui::Id::unique("ai-undo"));
    reg(&mut app.auto, "ai.undo", &undo, "Undo last edit");
    if undo.clicked() {
        act = Act::Undo;
    }
    lay.y += 30.0;

    // Preview of the last applied plan.
    if let Some(tex) = &app.ai_assistant.preview {
        let size = tex.size_vec2();
        let scale = (lay.w / size.x).min(180.0 / size.y).min(1.0);
        let r = Rect::from_min_size(pos2(lay.x0, lay.y), size * scale);
        ui.painter().image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), egui::Color32::WHITE);
        lay.y += r.height() + 8.0;
    }

    let a = sec_input(app, ui, &mut lay);
    if !matches!(a, Act::None) {
        act = a;
    }
    let a = sec_refs(app, ui, &mut lay);
    if !matches!(a, Act::None) {
        act = a;
    }
    let a = sec_settings(app, ui, &mut lay);
    if !matches!(a, Act::None) {
        act = a;
    }

    let scroll_end = lay.y - (rect.min.y + 12.0 - scroll.offset);
    scroll.end(ui, &mut app.auto, "ai.scroll", scroll_end, &t);
    ui.set_clip_rect(clip);
    let ctx = ui.ctx().clone();
    run_action(app, &ctx, act);
}
