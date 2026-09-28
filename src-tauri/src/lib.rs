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
            // Per-user app data dir, e.g. %APPDATA%/com.xarchlabs.mint on Windows.
            let data_dir = app
                .path()
                .app_data_dir()
                .map_err(|e| format!("no app data dir: {e}"))?
                .join("mint-memory");

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
