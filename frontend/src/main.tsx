import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";

import App from "./App";
import "./index.css";

// HashRouter keeps deep links working on any static host (GitHub Pages,
// Hugging Face Spaces) without server-side rewrite rules.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
