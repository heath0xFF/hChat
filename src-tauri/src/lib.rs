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
mod state;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
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
            commands::resolve_tool,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
