import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";
import fs from "fs";

// 开发模式：从 .server.token 读取 session token 并注入 HTML
const TOKEN_FILE = path.resolve(__dirname, "../.server.token");

function injectTokenPlugin() {
  return {
    name: "inject-session-token",
    transformIndexHtml(html: string) {
      let token = "";
      try {
        token = fs.readFileSync(TOKEN_FILE, "utf-8").trim();
      } catch {
        // token 文件不存在，跳过（后端未启动时）
      }
      if (token) {
        const script = `<script>window.__HERMES_SESSION_TOKEN__="${token}";</script>`;
        return html.replace("</head>", `${script}</head>`);
      }
      return html;
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), injectTokenPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "../hermes_cli/web_dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:9119",
    },
  },
});
