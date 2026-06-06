// Core modules are declared at the crate root via `#[path]` so the ported
// code's existing `crate::message::…` / `crate::config::…` references keep
// working unchanged. The files physically live under `src/core/`.
#[path = "core/message.rs"]
pub mod message;
#[path = "core/api.rs"]
pub mod api;
#[path = "core/storage.rs"]
pub mod storage;
#[path = "core/config.rs"]
pub mod config;
#[path = "core/slash.rs"]
pub mod slash;
#[path = "core/markdown.rs"]
pub mod markdown;
#[path = "core/tools.rs"]
pub mod tools;

mod commands;
mod metrics;
mod state;

use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .setup(|app| {
            // Spawn the metrics poller + macmon thread; expose the active-target
            // handle so commands can retarget it.
            let target = metrics::start(app.handle().clone());
            app.manage(metrics::MetricsHandle(target));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_config,
            commands::save_config,
            commands::fetch_models,
            commands::list_conversations,
            commands::load_conversation,
            commands::delete_conversation,
            commands::rename_conversation,
            commands::set_pinned,
            commands::search_conversations,
            commands::export_conversation,
            commands::cancel_stream,
            commands::send_message,
            commands::regenerate,
            commands::edit_message,
            commands::message_siblings,
            commands::walk_from,
            commands::resolve_tool,
            commands::list_presets,
            commands::create_preset,
            commands::delete_preset,
            commands::set_metrics_target,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
