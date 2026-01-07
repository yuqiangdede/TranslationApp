const $ = (id) => document.getElementById(id);

let ws = null;
let wsQueue = [];
let wsReconnectTimer = null;
let wsReconnectDelayMs = 500;
let currentMode = "file_in";
let lockCode = "";
let mic = null;
const AUDIO_MERGE_GAP_S = 1.2;
let audioMergeState = { mic_in: null, mic_out: null };
let asrPartialState = { mic_in: null, mic_out: null }; // { simple_id, in }
let uiMode = "debug"; // debug | simple
let holdState = { active: false, mode: null, prevMode: null };

function scheduleReconnect() {
  if (wsReconnectTimer) return;
  wsReconnectTimer = window.setTimeout(() => {
    wsReconnectTimer = null;
    connect().catch((e) => addSys(`连接失败：${e}`));
  }, wsReconnectDelayMs);
  wsReconnectDelayMs = Math.min(wsReconnectDelayMs * 2, 5000);
}

function wsSendJson(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify(obj));
      return true;
    } catch {}
  }
  wsQueue.push(obj);
  if (wsQueue.length > 200) wsQueue.shift();
  scheduleReconnect();
  return false;
}

function syncLockToServer() {
  wsSendJson({ type: "set_lock", code: lockCode, lock_mode: lockCode ? "manual" : "auto" });
}

function flushWsQueue() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  while (wsQueue.length) {
    const msg = wsQueue[0];
    try {
      ws.send(JSON.stringify(msg));
      wsQueue.shift();
    } catch {
      break;
    }
  }
}

function setStatus(text) {
  const t = String(text || "").trim();
  if (uiMode === "simple") {
    if (!t) {
      $("status").textContent = "状态：空闲";
      return;
    }
    if (t.includes("按住录音中")) {
      $("status").textContent = `状态：${t}`;
      return;
    }
    if (t.includes("已断开") || t.includes("连接错误") || t.includes("未连接")) {
      $("status").textContent = `状态：${t}`;
      return;
    }
    $("status").textContent = "状态：空闲";
    return;
  }
  $("status").textContent = `状态：${t}`;
}

function applySysVisibility() {
  document.body.classList.toggle("hide-sys", uiMode === "simple");
}

function applyUiMode() {
  document.body.classList.toggle("simple-mode", uiMode === "simple");
  const btn = $("btn_ui_mode");
  if (btn) btn.textContent = uiMode === "simple" ? "调试" : "简单";
  const btnLangAuto = $("btn_lang_auto");
  if (btnLangAuto) btnLangAuto.textContent = uiMode === "simple" ? "Auto" : "语种Auto";
  try {
    localStorage.setItem("uiMode", uiMode);
  } catch {}
  applySysVisibility();
  if (uiMode === "simple") setStatus("空闲");
  loadLanguages().catch(() => {});
  if (uiMode === "simple") setActiveMode("text_out");
  else if (!currentMode || currentMode === "text_out") setActiveMode("file_in");
}

function scrollToEnd() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function clearChat() {
  const chat = $("chat");
  chat.innerHTML = "";
  audioMergeState = { mic_in: null, mic_out: null };
  asrPartialState = { mic_in: null, mic_out: null };
  addSys("已清空。");
}

function setLangAuto() {
  lockCode = "";
  $("lock_select").value = "";
  setLockLabel();
  syncLockToServer();
  addSys("语种已切换为 Auto。");
}

function addSys(text) {
  if (uiMode === "simple") return;
  const div = document.createElement("div");
  div.className = "sys";
  div.textContent = text;
  $("chat").appendChild(div);
  scrollToEnd();
}

function addBubble({ side, text, ttsText }) {
  const row = document.createElement("div");
  row.className = `row ${side}`;

  const play = document.createElement("button");
  play.className = "play";
  play.textContent = "播放";
  play.dataset.tts = ttsText || "";
  play.disabled = !play.dataset.tts;
  play.addEventListener("click", () => {
    const t = play.dataset.tts || "";
    if (!t) return;
    wsSendJson({ type: "tts", text: t });
  });
  row.appendChild(play);

  const bubble = document.createElement("div");
  bubble.className = `bubble ${side === "left" ? "in" : "out"}`;
  bubble.textContent = text;
  row.appendChild(bubble);

  $("chat").appendChild(row);
  scrollToEnd();
  return { row, play, bubble };
}

function parseTtsText(displayText) {
  let t = (displayText || "").trim();
  if (!t) return "";
  t = t.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, "");
  const m = t.match(/^(IN|OUT)\b.*\):\s*(.*)$/);
  if (m) t = m[2].trim();
  t = t.replace(/^(ZH[:：]|中文[:：]|翻译[:：])\s*/i, "");
  t = t.replace(/（[^（）]{0,80}）\s*$/, "").trim();
  return t;
}

function renderAsrPartial(payload) {
  const mode = payload.mode || "";
  const simpleId = payload.simple_id || 0;
  const segEnd = typeof payload.seg_end_s === "number" ? payload.seg_end_s : 0;
  const ts = payload.ts || "";
  const langIn = payload.lang_in || "";
  const prob = typeof payload.prob === "number" ? payload.prob.toFixed(2) : "";
  const textIn = payload.text_in || "";
  const tAsr = payload.t_asr_ms || 0;

  if (!(mode === "mic_in" || mode === "mic_out")) return;
  if (!simpleId) return;
  if (!textIn) return;

  const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn} …`;
  const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
  const side = mode === "mic_out" ? "right" : "left";

  const st = asrPartialState[mode];
  if (st && st.simple_id === simpleId && st.in) {
    st.in.bubble.textContent = line1s;
    st.in.play.dataset.tts = "";
    st.in.play.disabled = true;
  } else {
    const b1 = addBubble({ side, text: line1s, ttsText: "" });
    b1.play.dataset.tts = "";
    b1.play.disabled = true;
    asrPartialState[mode] = { simple_id: simpleId, in: b1 };
  }

  const cur = asrPartialState[mode];
  if (!cur) return;
  if (mode === "mic_in") audioMergeState.mic_in = { lastEnd: segEnd || 0, in: cur.in, out: null };
  else audioMergeState.mic_out = { lastEnd: segEnd || 0, in: cur.in, out: null };
}

function finalizeAsrFromPartial(payload) {
  const mode = payload.mode || "";
  const simpleId = payload.simple_id || 0;
  const segEnd = typeof payload.seg_end_s === "number" ? payload.seg_end_s : 0;
  const ts = payload.ts || "";
  const langIn = payload.lang_in || "";
  const prob = typeof payload.prob === "number" ? payload.prob.toFixed(2) : "";
  const textIn = payload.text_in || "";
  const tAsr = payload.t_asr_ms || 0;

  if (!(mode === "mic_in" || mode === "mic_out")) return false;
  if (!simpleId) return false;

  const st = asrPartialState[mode];
  if (!st || st.simple_id !== simpleId || !st.in) return false;

  const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
  const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
  st.in.bubble.textContent = line1s;
  st.in.play.dataset.tts = textIn || "";
  st.in.play.disabled = !st.in.play.dataset.tts;

  if (mode === "mic_in") audioMergeState.mic_in = { lastEnd: segEnd || 0, in: st.in, out: null };
  else audioMergeState.mic_out = { lastEnd: segEnd || 0, in: st.in, out: null };
  asrPartialState[mode] = null;
  return true;
}

function renderAsr(payload) {
  if (payload && payload.simple_id && finalizeAsrFromPartial(payload)) return;
  const mode = payload.mode || "";
  const segStart = typeof payload.seg_start_s === "number" ? payload.seg_start_s : 0;
  const segEnd = typeof payload.seg_end_s === "number" ? payload.seg_end_s : 0;
  const ts = payload.ts || "";
  const langIn = payload.lang_in || "";
  const prob = typeof payload.prob === "number" ? payload.prob.toFixed(2) : "";
  const textIn = payload.text_in || "";
  const tAsr = payload.t_asr_ms || 0;

  if (mode === "mic_in") {
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    const st = audioMergeState.mic_in;
    if (st && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.in.bubble.textContent = `${st.in.bubble.textContent}\n${line1s}`.trim();
      st.in.play.dataset.tts = `${(st.in.play.dataset.tts || "").trim()} ${textIn}`.trim();
      st.in.play.disabled = !st.in.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b1 = addBubble({ side: "left", text: line1s, ttsText: textIn });
    audioMergeState.mic_in = { lastEnd: segEnd || 0, in: b1, out: null };
    return;
  }

  if (mode === "mic_out") {
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    const st = audioMergeState.mic_out;
    if (st && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.in.bubble.textContent = `${st.in.bubble.textContent}\n${line1s}`.trim();
      st.in.play.dataset.tts = `${(st.in.play.dataset.tts || "").trim()} ${textIn}`.trim();
      st.in.play.disabled = !st.in.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b1 = addBubble({ side: "right", text: line1s, ttsText: textIn });
    audioMergeState.mic_out = { lastEnd: segEnd || 0, in: b1, out: null };
    return;
  }

  if (mode === "file_in") {
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    addBubble({ side: "left", text: line1s, ttsText: parseTtsText(line1s) });
  }
}

function renderTr(payload) {
  const mode = payload.mode || "";
  const segStart = typeof payload.seg_start_s === "number" ? payload.seg_start_s : 0;
  const segEnd = typeof payload.seg_end_s === "number" ? payload.seg_end_s : 0;
  const ts = payload.ts || "";
  const dstCode = payload.dst_code || "";
  const textOut = payload.text_out || "";
  const tTr = payload.t_tr_ms || 0;

  if (mode === "mic_in") {
    const line2 = `[${ts}] ZH: ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    const st = audioMergeState.mic_in;
    if (st && st.out && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.out.bubble.textContent = `${st.out.bubble.textContent}\n${line2}`.trim();
      st.out.play.dataset.tts = `${(st.out.play.dataset.tts || "").trim()} ${textOut}`.trim();
      st.out.play.disabled = !st.out.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b2 = addBubble({ side: "left", text: line2, ttsText: textOut });
    if (st) {
      st.out = b2;
      st.lastEnd = segEnd || st.lastEnd;
    } else {
      audioMergeState.mic_in = { lastEnd: segEnd || 0, in: null, out: b2 };
    }
    return;
  }

  if (mode === "mic_out") {
    const dstLabel = dstCode ? dstCode : "L";
    const line2 = `[${ts}] OUT (${dstLabel}): ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    const st = audioMergeState.mic_out;
    if (st && st.out && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.out.bubble.textContent = `${st.out.bubble.textContent}\n${line2}`.trim();
      st.out.play.dataset.tts = `${(st.out.play.dataset.tts || "").trim()} ${textOut}`.trim();
      st.out.play.disabled = !st.out.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b2 = addBubble({ side: "right", text: line2, ttsText: textOut });
    if (st) {
      st.out = b2;
      st.lastEnd = segEnd || st.lastEnd;
    } else {
      audioMergeState.mic_out = { lastEnd: segEnd || 0, in: null, out: b2 };
    }
    return;
  }

  if (mode === "file_in") {
    const line2 = `[${ts}] ZH: ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    addBubble({ side: "left", text: line2, ttsText: textOut });
    addSys("");
  }
}

function renderSegment(payload) {
  const mode = payload.mode || "";
  const segStart = typeof payload.seg_start_s === "number" ? payload.seg_start_s : 0;
  const segEnd = typeof payload.seg_end_s === "number" ? payload.seg_end_s : 0;
  const ts = payload.ts || "";
  const langIn = payload.lang_in || "";
  const prob = typeof payload.prob === "number" ? payload.prob.toFixed(2) : "";
  const textIn = payload.text_in || "";
  const dstCode = payload.dst_code || "";
  const textOut = payload.text_out || "";
  const tAsr = payload.t_asr_ms || 0;
  const tTr = payload.t_tr_ms || 0;

  if (mode === "mic_in") {
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    const line2 = `[${ts}] ZH: ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    const st = audioMergeState.mic_in;
    if (st && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.in.bubble.textContent = `${st.in.bubble.textContent}\n${line1s}`.trim();
      st.in.play.dataset.tts = `${(st.in.play.dataset.tts || "").trim()} ${textIn}`.trim();
      st.in.play.disabled = !st.in.play.dataset.tts;

      st.out.bubble.textContent = `${st.out.bubble.textContent}\n${line2}`.trim();
      st.out.play.dataset.tts = `${(st.out.play.dataset.tts || "").trim()} ${textOut}`.trim();
      st.out.play.disabled = !st.out.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b1 = addBubble({ side: "left", text: line1s, ttsText: textIn });
    const b2 = addBubble({ side: "left", text: line2, ttsText: textOut });
    audioMergeState.mic_in = { lastEnd: segEnd || 0, in: b1, out: b2 };
    return;
  }

  if (mode === "mic_out") {
    const dstLabel = dstCode ? dstCode : "L";
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    const line2 = `[${ts}] OUT (${dstLabel}): ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    const st = audioMergeState.mic_out;
    if (st && segStart - st.lastEnd <= AUDIO_MERGE_GAP_S) {
      st.in.bubble.textContent = `${st.in.bubble.textContent}\n${line1s}`.trim();
      st.in.play.dataset.tts = `${(st.in.play.dataset.tts || "").trim()} ${textIn}`.trim();
      st.in.play.disabled = !st.in.play.dataset.tts;

      st.out.bubble.textContent = `${st.out.bubble.textContent}\n${line2}`.trim();
      st.out.play.dataset.tts = `${(st.out.play.dataset.tts || "").trim()} ${textOut}`.trim();
      st.out.play.disabled = !st.out.play.dataset.tts;
      st.lastEnd = segEnd || st.lastEnd;
      return;
    }
    const b1 = addBubble({ side: "right", text: line1s, ttsText: textIn });
    const b2 = addBubble({ side: "right", text: line2, ttsText: textOut });
    audioMergeState.mic_out = { lastEnd: segEnd || 0, in: b1, out: b2 };
    return;
  }

  if (mode === "file_in") {
    const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
    const line1s = tAsr ? `${line1}（识别 ${tAsr.toFixed(0)}ms）` : line1;
    const line2 = `[${ts}] ZH: ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
    addBubble({ side: "left", text: line1s, ttsText: parseTtsText(line1s) });
    addBubble({ side: "left", text: line2, ttsText: textOut });
    addSys("");
    return;
  }

  const dstLabel = dstCode ? dstCode : "L";
  const line1 = `[${ts}] IN (${langIn} ${prob}): ${textIn}`;
  const line2 = `[${ts}] OUT (${dstLabel}): ${textOut}（翻译 ${tTr.toFixed(0)}ms）`;
  addBubble({ side: "right", text: line1, ttsText: parseTtsText(line1) });
  addBubble({ side: "right", text: line2, ttsText: textOut });
  addSys("");
}

async function loadLanguages() {
  const resp = await fetch("/api/languages");
  const data = await resp.json();
  const sel = $("lock_select");
  sel.innerHTML = "";
  const optAuto = document.createElement("option");
  optAuto.value = "";
  optAuto.textContent = "Auto";
  sel.appendChild(optAuto);

  const isSimple = uiMode === "simple";
  const stableGroup = document.createElement("optgroup");
  stableGroup.label = "可用";
  const expGroup = document.createElement("optgroup");
  expGroup.label = "实验";

  for (const c of data.choices || []) {
    if (isSimple && !c.stable) continue;
    const opt = document.createElement("option");
    opt.value = c.code;
    const label = c.label || c.code;
    const shortLabel = String(label).split("（")[0].split("(")[0].trim() || label;
    if (c.stable) {
      opt.textContent = isSimple ? shortLabel : label;
      opt.classList.add("stable");
      if (isSimple) sel.appendChild(opt);
      else stableGroup.appendChild(opt);
    } else {
      opt.textContent = `${label}（实验）`;
      expGroup.appendChild(opt);
    }
  }
  if (!isSimple) {
    if (stableGroup.children.length) sel.appendChild(stableGroup);
    if (expGroup.children.length) sel.appendChild(expGroup);
  }
}

function setLockLabel() {
  const el = $("lock_label");
  if (!el) return;
  el.textContent = lockCode ? `语种：${lockCode}` : "语种：Auto";
}

function setActiveMode(mode) {
  currentMode = mode;
  audioMergeState = { mic_in: null, mic_out: null };
  for (const id of ["btn_mic_in", "btn_file_in", "btn_mic_out", "btn_text_out"]) {
    $(id).classList.toggle("active", id.endsWith(mode));
  }
  const enableText = mode === "text_out";
  $("text_input").disabled = !enableText;
  $("btn_send").disabled = !enableText;
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/api/file_in_stream", { method: "POST", body: fd });
  const ct = (resp.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    const data = await resp.json();
    if (!data.ok) addSys(`ERROR: ${data.error || "上传失败"}`);
    return;
  }
  if (!resp.ok || !resp.body) {
    addSys(`ERROR: 上传失败（HTTP ${resp.status}）。`);
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    while (true) {
      const idx = buf.indexOf("\n");
      if (idx < 0) break;
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        const ev = JSON.parse(line);
        onEvent(ev);
      } catch {}
    }
  }
}

function onEvent(ev) {
  if (!ev) return;
  const fromWs = ev.type === "event";
  const kind = ev.kind;
  const payload = ev.payload || {};
  if (kind === "status") {
    const t = String(payload.text || "").trim();
    if (uiMode === "simple") {
      if (!(t.includes("按住录音中") || t.includes("已断开") || t.includes("连接错误") || t.includes("未连接"))) return;
    }
    setStatus(t);
    addSys(t);
  } else if (kind === "error") {
    setStatus(payload.text || "");
    addSys(payload.text || "");
  } else if (kind === "mode") {
    const mode = payload.mode || "";
    if (mode) {
      if (uiMode === "simple") return;
      // 服务端推送的模式切换：仅更新 UI，并在离开音频模式时释放麦克风。
      const prev = currentMode;
      setActiveMode(mode);
      if (!(mode === "mic_in" || mode === "mic_out") && (prev === "mic_in" || prev === "mic_out")) stopMic();
    }
  } else if (kind === "lock_lang") {
    lockCode = payload.code || "";
    $("lock_select").value = lockCode;
    setLockLabel();
    addSys(`语种 L 已锁定：${lockCode}`);
    // file_in 的 lock_lang 事件来自 HTTP 返回，不会同步到 WebSocket Session；
    // 这里显式同步一次，避免 text_out 提示“尚未锁定语种 L”。
    if (!fromWs) syncLockToServer();
  } else if (kind === "segment") {
    const ts = new Date((ev.ts || Date.now()) * 1000).toTimeString().slice(0, 8);
    renderSegment({ ...payload, ts });
  } else if (kind === "asr") {
    const ts = new Date((ev.ts || Date.now()) * 1000).toTimeString().slice(0, 8);
    renderAsr({ ...payload, ts });
  } else if (kind === "asr_partial") {
    const ts = new Date((ev.ts || Date.now()) * 1000).toTimeString().slice(0, 8);
    renderAsrPartial({ ...payload, ts });
  } else if (kind === "tr") {
    const ts = new Date((ev.ts || Date.now()) * 1000).toTimeString().slice(0, 8);
    renderTr({ ...payload, ts });
  } else if (kind === "file_done") {
    if (payload && payload.ok === false) addSys(`ERROR: ${payload.error || "文件处理失败"}`);
  }
}

async function openSettings() {
  const resp = await fetch("/api/config");
  const data = await resp.json();
  $("config_text").value = JSON.stringify(data.config || {}, null, 2);
  $("dlg_settings").showModal();
}

async function saveSettings() {
  let cfg = null;
  try {
    cfg = JSON.parse($("config_text").value);
  } catch {
    addSys("配置 JSON 解析失败。");
    return;
  }
  const resp = await fetch("/api/config", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(cfg),
  });
  const data = await resp.json();
  if (!data.ok) {
    addSys(`保存失败：${data.error || ""}`);
    return;
  }
  addSys("配置已保存。");
  wsSendJson({ type: "reload_config" });
  $("dlg_settings").close();
}

async function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  await loadLanguages();
  setLockLabel();

  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${wsProto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus(uiMode === "simple" ? "空闲" : "已连接");
    wsReconnectDelayMs = 500;
    // 连接后先同步模式/语种，再发送积压消息。
    if (uiMode !== "simple") wsSendJson({ type: "set_mode", mode: currentMode });
    syncLockToServer();
    flushWsQueue();
  };
  ws.onclose = () => {
    setStatus("已断开");
    scheduleReconnect();
  };
  ws.onerror = () => {
    setStatus("连接错误");
    scheduleReconnect();
  };

  ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "hello") {
          // 不用 hello 覆盖本地选择（可能已经被 file_in 结果/用户选择更新）。
          if (uiMode !== "simple") setActiveMode(currentMode || msg.mode || "mic_in");
          if (!lockCode) lockCode = (msg.lock && msg.lock.code) || "";
          $("lock_select").value = lockCode;
          setLockLabel();
          if (uiMode !== "simple") wsSendJson({ type: "set_mode", mode: currentMode });
          syncLockToServer();
          return;
        }
      if (msg.type === "event") {
        onEvent(msg);
      }
    }
  };
}

function downsampleTo16k(float32, inputRate) {
  if (inputRate === 16000) return float32;
  const ratio = inputRate / 16000;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = float32[idx] || 0;
    const b = float32[idx + 1] || a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function floatToPcm16(float32) {
  const buf = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

async function startMic() {
  if (mic) return;
  if (!window.isSecureContext) {
    throw new Error("浏览器限制：非安全上下文无法使用麦克风（请用 https 或 localhost/127.0.0.1）。");
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("浏览器不支持麦克风 API（getUserMedia）。");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!(currentMode === "mic_in" || currentMode === "mic_out")) return;
    const input = ev.inputBuffer.getChannelData(0);
    const ds = downsampleTo16k(input, ctx.sampleRate);
    ws.send(floatToPcm16(ds));
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  mic = { stream, ctx, src, proc };
}

function stopMic() {
  if (!mic) return;
  mic.proc.disconnect();
  mic.src.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close();
  mic = null;
}

function setMode(mode) {
  if (uiMode === "simple") return;
  setActiveMode(mode);
  wsSendJson({ type: "set_mode", mode });
  if (mode === "mic_in" || mode === "mic_out") startMic().catch((e) => addSys(`麦克风失败：${e}`));
  else stopMic();
}

function holdStart(mode) {
  if (holdState.active) return;
  if (!(mode === "mic_in" || mode === "mic_out")) return;
  holdState.active = true;
  holdState.mode = mode;
  holdState.prevMode = currentMode;
  setActiveMode(mode);
  setStatus(`按住录音中：${mode} ...`);
  wsSendJson({ type: "simple_start", mode });
  startMic().catch((e) => addSys(`麦克风失败：${e}`));
}

function holdEnd() {
  if (!holdState.active) return;
  holdState.active = false;
  wsSendJson({ type: "simple_end" });
  stopMic();
  const back = holdState.prevMode || "text_out";
  holdState.mode = null;
  holdState.prevMode = null;
  setActiveMode(back);
  setStatus("空闲");
}

window.addEventListener("load", () => {
  try {
    uiMode = localStorage.getItem("uiMode") || "debug";
  } catch {}
  if (uiMode !== "simple") uiMode = "debug";
  applyUiMode();
  connect().catch((e) => addSys(String(e)));

  $("btn_mic_in").addEventListener("click", () => setMode("mic_in"));
  $("btn_mic_out").addEventListener("click", () => setMode("mic_out"));
  $("btn_text_out").addEventListener("click", () => setMode("text_out"));
  $("btn_file_in").addEventListener("click", () => {
    setMode("file_in");
    $("file_picker").click();
  });

  $("btn_ui_mode").addEventListener("click", () => {
    uiMode = uiMode === "simple" ? "debug" : "simple";
    applyUiMode();
    if (uiMode === "simple") {
      stopMic();
      wsSendJson({ type: "set_mode", mode: "text_out" });
      setStatus("空闲");
    } else {
      wsSendJson({ type: "set_mode", mode: currentMode });
      if (currentMode === "mic_in" || currentMode === "mic_out") startMic().catch((e) => addSys(`麦克风失败：${e}`));
    }
  });

  const bindHold = (id, mode) => {
    const el = $(id);
    if (!el) return;
    // 手机端长按会触发“右键/菜单”，这里屏蔽掉，避免打断按住录音交互。
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      return false;
    });
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      holdStart(mode);
    });
    const end = (e) => {
      e.preventDefault();
      holdEnd();
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("pointerleave", (e) => {
      if (!holdState.active) return;
      end(e);
    });
  };
  bindHold("btn_hold_listen", "mic_in");
  bindHold("btn_hold_speak", "mic_out");

  $("file_picker").addEventListener("change", async () => {
    const f = $("file_picker").files && $("file_picker").files[0];
    if (f) await uploadFile(f);
    $("file_picker").value = "";
  });

  $("lock_select").addEventListener("change", () => {
    lockCode = $("lock_select").value;
    setLockLabel();
    syncLockToServer();
  });

  $("btn_clear").addEventListener("click", () => clearChat());
  $("btn_lang_auto").addEventListener("click", () => setLangAuto());

  $("btn_send").addEventListener("click", () => {
    if (uiMode === "simple") return;
    const t = $("text_input").value.trim();
    if (!t) return;
    syncLockToServer();
    const ok = wsSendJson({ type: "text_out", text: t });
    if (!ok) addSys("WebSocket 未连接，消息已加入队列，正在重连…");
    $("text_input").value = "";
  });

  $("text_input").addEventListener("keydown", (e) => {
    if (uiMode === "simple") return;
    if (e.key === "Enter") $("btn_send").click();
  });

  $("btn_settings").addEventListener("click", () => openSettings().catch((e) => addSys(String(e))));
  $("btn_close_settings").addEventListener("click", () => $("dlg_settings").close());
  $("btn_save_cfg").addEventListener("click", () => saveSettings().catch((e) => addSys(String(e))));
  $("btn_reload_cfg").addEventListener("click", () => wsSendJson({ type: "reload_config" }));
});
