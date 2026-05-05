/**
 * chatClient：主线程侧的薄封装。
 *
 * - 优先用 SharedWorker：多 Tab/刷新都能续上同一条流；
 * - 不支持 SharedWorker 时降级到普通 Worker（仍可避免主线程被阻塞，但失去多 Tab 共享）；
 * - Worker 都不支持时降级到主线程 fetch（最低保障，但刷新就丢）。
 *
 * 用法：
 *   const client = getChatClient();
 *   const off = client.subscribe((msg) => { ... });
 *   client.send({ message, sessionId, messageId });
 *   client.resume(messageId, lastSeq);
 *   client.abort(messageId);
 */

export type ChatToolEvent = {
  seq: number;
  kind: "tool_started" | "tool_completed";
  toolName: string;
  preview?: string;
  args?: Record<string, unknown>;
  result?: string;
  duration?: number;
  isError?: boolean;
};

export type ChatEvent =
  | { type: "meta"; messageId: string; sessionId: string; resume?: boolean }
  | { type: "token"; messageId: string; seq: number; delta: string }
  | {
      type: "tool_started";
      messageId: string;
      seq: number;
      toolName: string;
      preview?: string;
      args?: Record<string, unknown>;
    }
  | {
      type: "tool_completed";
      messageId: string;
      seq: number;
      toolName: string;
      preview?: string;
      args?: Record<string, unknown>;
      result?: string;
      duration?: number;
      isError?: boolean;
    }
  | { type: "done"; messageId: string; final: string }
  | { type: "error"; messageId: string; error: string }
  | {
      type: "snapshot";
      streams: {
        messageId: string;
        sessionId: string;
        status: "running" | "done" | "error" | "aborted";
        lastSeq: number;
        text: string;
        finalText?: string;
        error?: string;
        timeline?: ChatToolEvent[];
      }[];
    };

type Listener = (msg: ChatEvent) => void;

interface ChatClient {
  subscribe(fn: Listener): () => void;
  send(args: { message: string; sessionId?: string; messageId: string; token?: string }): void;
  resume(args: { messageId: string; lastSeq: number; token?: string }): void;
  abort(messageId: string): void;
  snapshot(): void;
}

let _singleton: ChatClient | null = null;

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
  }
}

function readToken(): string | undefined {
  return typeof window !== "undefined" ? window.__HERMES_SESSION_TOKEN__ : undefined;
}

/**
 * SharedWorker 实现
 */
function buildSharedWorkerClient(): ChatClient {
  const worker = new SharedWorker(
    new URL("../workers/chatSharedWorker.ts", import.meta.url),
    { type: "module", name: "hermes-chat" },
  );
  const port = worker.port;
  port.start();

  const listeners = new Set<Listener>();
  port.onmessage = (e: MessageEvent) => {
    for (const l of listeners) l(e.data as ChatEvent);
  };

  // 关闭 Tab 时不主动 abort 流（强接续的关键）
  port.postMessage({ type: "subscribe" });

  return {
    subscribe(fn) {
      listeners.add(fn);
      // 新订阅者立刻请求一份快照，把当前进行中的 stream 同步过来
      port.postMessage({ type: "snapshot" });
      return () => listeners.delete(fn);
    },
    send(args) {
      port.postMessage({ ...args, token: args.token ?? readToken(), type: "send" });
    },
    resume(args) {
      port.postMessage({ ...args, token: args.token ?? readToken(), type: "resume" });
    },
    abort(messageId) {
      port.postMessage({ type: "abort", messageId });
    },
    snapshot() {
      port.postMessage({ type: "snapshot" });
    },
  };
}

/**
 * 主线程降级：直接 fetch，不持久化、不共享，仅保证最基础可用
 */
function buildMainThreadClient(): ChatClient {
  const listeners = new Set<Listener>();
  const aborts = new Map<string, AbortController>();

  async function pump(messageId: string, resp: Response) {
    if (!resp.ok || !resp.body) {
      const err = `HTTP ${resp.status}`;
      for (const l of listeners) l({ type: "error", messageId, error: err });
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        let data = "";
        let id: string | undefined;
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7).trim();
          else if (line.startsWith("data: ")) data += (data ? "\n" : "") + line.slice(6);
          else if (line.startsWith("id: ")) id = line.slice(4).trim();
        }
        if (!data) continue;
        let payload: any = {};
        try {
          payload = JSON.parse(data);
        } catch {
          /* ignore */
        }
        if (event === "meta") {
          for (const l of listeners)
            l({ type: "meta", messageId, sessionId: payload.session_id, resume: !!payload.resume });
        } else if (event === "token") {
          const seq = id != null ? Number(id) : 0;
          for (const l of listeners) l({ type: "token", messageId, seq, delta: payload.delta ?? "" });
        } else if (event === "tool_started") {
          const seq = id != null ? Number(id) : 0;
          for (const l of listeners) {
            l({
              type: "tool_started",
              messageId,
              seq,
              toolName: String(payload.tool_name ?? ""),
              preview: typeof payload.preview === "string" ? payload.preview : undefined,
              args: (payload.args as Record<string, unknown> | undefined) ?? undefined,
            });
          }
        } else if (event === "tool_completed") {
          const seq = id != null ? Number(id) : 0;
          for (const l of listeners) {
            l({
              type: "tool_completed",
              messageId,
              seq,
              toolName: String(payload.tool_name ?? ""),
              preview: typeof payload.preview === "string" ? payload.preview : undefined,
              args: (payload.args as Record<string, unknown> | undefined) ?? undefined,
              result: typeof payload.result === "string" ? payload.result : undefined,
              duration: typeof payload.duration === "number" ? payload.duration : undefined,
              isError: payload.is_error === true,
            });
          }
        } else if (event === "done") {
          for (const l of listeners) l({ type: "done", messageId, final: payload.final ?? "" });
        } else if (event === "error") {
          for (const l of listeners) l({ type: "error", messageId, error: String(payload.error ?? "") });
        }
      }
    }
  }

  return {
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    async send(args) {
      const ctl = new AbortController();
      aborts.set(args.messageId, ctl);
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      const tk = args.token ?? readToken();
      if (tk) headers.Authorization = `Bearer ${tk}`;
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: args.message,
          session_id: args.sessionId,
          message_id: args.messageId,
        }),
        signal: ctl.signal,
      });
      await pump(args.messageId, resp);
    },
    async resume(args) {
      const ctl = new AbortController();
      aborts.set(args.messageId, ctl);
      const headers: Record<string, string> = {};
      const tk = args.token ?? readToken();
      if (tk) headers.Authorization = `Bearer ${tk}`;
      const url = `/api/chat/resume?message_id=${encodeURIComponent(args.messageId)}&last_seq=${args.lastSeq}`;
      const resp = await fetch(url, { method: "GET", headers, signal: ctl.signal });
      await pump(args.messageId, resp);
    },
    abort(messageId) {
      aborts.get(messageId)?.abort();
    },
    snapshot() {
      // 主线程模式没有跨 Tab 状态，无快照可发
    },
  };
}

export function getChatClient(): ChatClient {
  if (_singleton) return _singleton;
  try {
    if (typeof SharedWorker !== "undefined") {
      _singleton = buildSharedWorkerClient();
      return _singleton;
    }
  } catch {
    /* fallthrough */
  }
  _singleton = buildMainThreadClient();
  return _singleton;
}

/**
 * 在 IndexedDB 里查"上次未完成的 messageId"，方便页面挂载时自动续传。
 * 仅 SharedWorker 模式下有效（worker 内会写）。
 */
export async function findPendingMessage(): Promise<
  { messageId: string; sessionId: string; lastSeq: number } | null
> {
  if (typeof indexedDB === "undefined") return null;
  return new Promise((resolve) => {
    const req = indexedDB.open("hermes-chat-stream", 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore("pending", { keyPath: "messageId" });
    };
    req.onsuccess = () => {
      const db = req.result;
      try {
        const tx = db.transaction("pending", "readonly");
        const all = tx.objectStore("pending").getAll();
        all.onsuccess = () => {
          const list = all.result as Array<{ messageId: string; sessionId: string; lastSeq: number }>;
          db.close();
          resolve(list.length ? list[list.length - 1] : null);
        };
        all.onerror = () => {
          db.close();
          resolve(null);
        };
      } catch {
        db.close();
        resolve(null);
      }
    };
    req.onerror = () => resolve(null);
  });
}
