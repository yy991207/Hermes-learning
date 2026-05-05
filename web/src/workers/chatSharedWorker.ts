/**
 * Chat SharedWorker
 *
 * 角色：在浏览器进程里以单实例形式持有"和后端 /api/chat/stream 的 SSE 长连接"。
 * 多 Tab 通过 MessagePort 共享同一份流；页面刷新不会断流。
 *
 * 协议（主线程 -> worker）：
 *   { type: 'subscribe' }
 *   { type: 'send',   sessionId?, message, messageId, token? }
 *   { type: 'resume', messageId, lastSeq, token? }
 *   { type: 'abort',  messageId }
 *   { type: 'snapshot' }                       // 询问当前所有 stream 的快照
 *
 * 协议（worker -> 主线程）：
 *   { type: 'meta',   messageId, sessionId, resume? }
 *   { type: 'token',  messageId, seq, delta }
 *   { type: 'done',   messageId, final }
 *   { type: 'error',  messageId, error }
 *   { type: 'snapshot', streams: { messageId, sessionId, status, lastSeq, text }[] }
 */

interface StreamToolEvent {
  seq: number;
  kind: "tool_started" | "tool_completed";
  toolName: string;
  preview?: string;
  args?: Record<string, unknown>;
  result?: string;
  duration?: number;
  isError?: boolean;
}

interface StreamState {
  messageId: string;
  sessionId: string;
  status: "running" | "done" | "error" | "aborted";
  buffer: { seq: number; delta: string }[];
  text: string;          // 累积文本，方便新 Tab 一次性回放
  timeline: StreamToolEvent[];
  finalText?: string;
  error?: string;
  abort?: AbortController;
}

const ports = new Set<MessagePort>();
const streams = new Map<string, StreamState>();

// IndexedDB：存"未完成的 messageId + lastSeq"，用于 SharedWorker 自身被回收后的恢复
const DB_NAME = "hermes-chat-stream";
const STORE = "pending";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: "messageId" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbPut(record: { messageId: string; sessionId: string; lastSeq: number; status: string }) {
  try {
    const db = await openDB();
    await new Promise<void>((res, rej) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(record);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
    db.close();
  } catch {
    // 静默：IDB 失败不影响主流程
  }
}

async function dbDelete(messageId: string) {
  try {
    const db = await openDB();
    await new Promise<void>((res, rej) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(messageId);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
    db.close();
  } catch {
    /* ignore */
  }
}

function broadcast(msg: unknown) {
  for (const p of ports) {
    try {
      p.postMessage(msg);
    } catch {
      ports.delete(p);
    }
  }
}

function lastSeqOf(s: StreamState): number {
  return s.buffer.length === 0 ? -1 : s.buffer[s.buffer.length - 1].seq;
}

/**
 * 解析 SSE 字节流（适用于 fetch + ReadableStream）
 * 通用解析器：每收到一个完整事件就触发回调
 */
async function pumpSSE(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (ev: { event: string; data: string; id?: string }) => void,
) {
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
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
      if (data) onEvent({ event, data, id });
    }
  }
}

/**
 * 处理一帧 SSE 事件：写 buffer + 广播 + 落 IDB
 */
function handleFrame(state: StreamState, ev: { event: string; data: string; id?: string }) {
  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(ev.data);
  } catch {
    return;
  }

  if (ev.event === "meta") {
    if (typeof payload.session_id === "string") state.sessionId = payload.session_id;
    broadcast({
      type: "meta",
      messageId: state.messageId,
      sessionId: state.sessionId,
      resume: payload.resume === true,
    });
  } else if (ev.event === "token") {
    const seq = ev.id != null ? Number(ev.id) : state.buffer.length;
    const delta = (payload.delta as string) ?? "";
    state.buffer.push({ seq, delta });
    state.text += delta;
    broadcast({ type: "token", messageId: state.messageId, seq, delta });
    // 节流落 IDB：每 10 个 chunk 落一次
    if (seq % 10 === 0) {
      dbPut({
        messageId: state.messageId,
        sessionId: state.sessionId,
        lastSeq: seq,
        status: "running",
      });
    }
  } else if (ev.event === "tool_started") {
    const seq = ev.id != null ? Number(ev.id) : state.buffer.length;
    const item: StreamToolEvent = {
      seq,
      kind: "tool_started",
      toolName: String(payload.tool_name ?? ""),
      preview: typeof payload.preview === "string" ? payload.preview : undefined,
      args: (payload.args as Record<string, unknown> | undefined) ?? undefined,
    };
    state.timeline.push(item);
    broadcast({
      type: "tool_started",
      messageId: state.messageId,
      seq,
      toolName: item.toolName,
      preview: item.preview,
      args: item.args,
    });
  } else if (ev.event === "tool_completed") {
    const seq = ev.id != null ? Number(ev.id) : state.buffer.length;
    const item: StreamToolEvent = {
      seq,
      kind: "tool_completed",
      toolName: String(payload.tool_name ?? ""),
      preview: typeof payload.preview === "string" ? payload.preview : undefined,
      args: (payload.args as Record<string, unknown> | undefined) ?? undefined,
      result: typeof payload.result === "string" ? payload.result : undefined,
      duration: typeof payload.duration === "number" ? payload.duration : undefined,
      isError: payload.is_error === true,
    };
    state.timeline.push(item);
    broadcast({
      type: "tool_completed",
      messageId: state.messageId,
      seq,
      toolName: item.toolName,
      preview: item.preview,
      args: item.args,
      result: item.result,
      duration: item.duration,
      isError: item.isError,
    });
  } else if (ev.event === "done") {
    state.status = "done";
    state.finalText = (payload.final as string) ?? state.text;
    broadcast({ type: "done", messageId: state.messageId, final: state.finalText });
    dbDelete(state.messageId);
  } else if (ev.event === "error") {
    state.status = "error";
    state.error = String(payload.error ?? "unknown");
    broadcast({ type: "error", messageId: state.messageId, error: state.error });
    dbDelete(state.messageId);
  }
}

async function startSend(args: {
  message: string;
  messageId: string;
  sessionId?: string;
  token?: string;
}) {
  const state: StreamState = {
    messageId: args.messageId,
    sessionId: args.sessionId ?? "",
    status: "running",
    buffer: [],
    text: "",
    timeline: [],
    abort: new AbortController(),
  };
  streams.set(args.messageId, state);

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (args.token) headers["Authorization"] = `Bearer ${args.token}`;

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body: JSON.stringify({
        message: args.message,
        session_id: args.sessionId,
        message_id: args.messageId,
      }),
      signal: state.abort?.signal,
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const reader = resp.body.getReader();
    await pumpSSE(reader, (ev) => handleFrame(state, ev));
    if (state.status === "running") {
      // 连接结束但没收到 done/error：当作 error
      state.status = "error";
      state.error = "stream closed unexpectedly";
      broadcast({ type: "error", messageId: state.messageId, error: state.error });
    }
  } catch (e) {
    state.status = "error";
    state.error = String(e);
    broadcast({ type: "error", messageId: state.messageId, error: state.error });
  }
}

async function startResume(args: { messageId: string; lastSeq: number; token?: string }) {
  let state = streams.get(args.messageId);
  if (!state) {
    state = {
      messageId: args.messageId,
      sessionId: "",
      status: "running",
      buffer: [],
      text: "",
      timeline: [],
      abort: new AbortController(),
    };
    streams.set(args.messageId, state);
  }
  const headers: Record<string, string> = {};
  if (args.token) headers["Authorization"] = `Bearer ${args.token}`;
  const url = `/api/chat/resume?message_id=${encodeURIComponent(args.messageId)}&last_seq=${args.lastSeq}`;
  try {
    const resp = await fetch(url, { method: "GET", headers, signal: state.abort?.signal });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    await pumpSSE(reader, (ev) => handleFrame(state!, ev));
  } catch (e) {
    state.status = "error";
    state.error = String(e);
    broadcast({ type: "error", messageId: state.messageId, error: state.error });
  }
}

function snapshotForPort(port: MessagePort) {
  const arr = Array.from(streams.values()).map((s) => ({
    messageId: s.messageId,
    sessionId: s.sessionId,
    status: s.status,
    lastSeq: lastSeqOf(s),
    text: s.text,
    finalText: s.finalText,
    error: s.error,
    timeline: s.timeline,
  }));
  port.postMessage({ type: "snapshot", streams: arr });
}

function handlePortMessage(port: MessagePort, msg: any) {
  if (!msg || typeof msg !== "object") return;
  switch (msg.type) {
    case "subscribe":
      // 新 Tab 接入，立刻给一份快照，避免它错过已发生的事件
      snapshotForPort(port);
      break;
    case "send":
      startSend({
        message: msg.message,
        messageId: msg.messageId,
        sessionId: msg.sessionId,
        token: msg.token,
      });
      break;
    case "resume":
      startResume({ messageId: msg.messageId, lastSeq: msg.lastSeq ?? -1, token: msg.token });
      break;
    case "abort": {
      const s = streams.get(msg.messageId);
      s?.abort?.abort();
      break;
    }
    case "snapshot":
      snapshotForPort(port);
      break;
    default:
      break;
  }
}

// SharedWorker 入口
// TS 在 SharedWorker scope 下缺少 SharedWorkerGlobalScope 类型声明，这里手动补上
declare interface SharedWorkerGlobalScope {
  onconnect: ((this: SharedWorkerGlobalScope, ev: MessageEvent) => void) | null;
}
(self as unknown as SharedWorkerGlobalScope).onconnect = (e: MessageEvent) => {
  const port = (e as unknown as { ports: MessagePort[] }).ports[0];
  ports.add(port);
  port.onmessage = (ev: MessageEvent) => handlePortMessage(port, ev.data);
  port.start();
};

export {}; // 让 TS 把它当模块
