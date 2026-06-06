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
#[path = "core/agents.rs"]
pub mod agents;

mod commands;
mod mcp;
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

            // Connect MCP servers in the background.
            let state = app.state::<AppState>();

            // Enforce the usage retention window once at startup.
            let retention = state.config.lock().unwrap().usage_retention_days;
            if retention > 0 {
                state.storage.lock().unwrap().prune_usage(retention);
            }

            let mcp = state.mcp.clone();
            let servers = state.config.lock().unwrap().mcp_servers.clone();
            if !servers.is_empty() {
                tauri::async_runtime::spawn(async move {
                    mcp.connect_all(servers).await;
                });
            }
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
            commands::usage_stats,
            commands::clear_usage,
            commands::run_benchmark,
            commands::list_projects,
            commands::create_project,
            commands::rename_project,
            commands::delete_project,
            commands::set_project_pinned,
            commands::set_conversation_project,
            commands::search_conversations,
            commands::export_conversation,
            commands::export_conversation_file,
            commands::save_draft,
            commands::list_agents,
            commands::list_mcp_servers,
            commands::reconnect_mcp,
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
