mod commands;
mod memory;

use std::sync::Mutex;

use tauri::Manager;

use commands::AppState;
use memory::MemoryEngine;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Where the on-device memory lives. `MINT_DATA_DIR` overrides the
            // default (used in dev to keep the store inside the project, easy to
            // inspect); otherwise fall back to the per-user app data dir.
            let data_dir = match std::env::var("MINT_DATA_DIR") {
                Ok(dir) if !dir.trim().is_empty() => std::path::PathBuf::from(dir),
                _ => app
                    .path()
                    .app_data_dir()
                    .map_err(|e| format!("no app data dir: {e}"))?
                    .join("mint-memory"),
            };
            log::info!("Mint data dir: {}", data_dir.display());

            let engine = MemoryEngine::open(&data_dir).map_err(|e| e.to_string())?;
            app.manage(AppState {
                engine: Mutex::new(engine),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::add_memory,
            commands::search_memories,
            commands::list_memories,
            commands::delete_memory,
            commands::stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
