// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Xarch Labs
import ReactDOM from "react-dom/client";
import App from "./App";

// StrictMode intentionally omitted: its dev-only double-render/double-effect
// doubles work (and previously double-registered Tauri listeners), which hurt
// perceived performance in this webview.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(<App />);
