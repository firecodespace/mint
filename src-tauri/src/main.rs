// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mint_lib::run()
}
