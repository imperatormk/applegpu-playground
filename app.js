const API = (window.PLAYGROUND_API || "").replace(/\/$/, "");
const $ = s => document.querySelector(s);

const KERNELS = window.KERNELS || [];
const zoo = $("#zoo");
KERNELS.forEach((k, i) => {
  const opt = document.createElement("option");
  opt.value = String(i);
  opt.textContent = k.label;
  zoo.appendChild(opt);
});

let editor = null;
const initialSource = KERNELS.length ? KERNELS[0].source : "";
const getSource = () => (editor ? editor.getValue() : initialSource);
const setSource = s => { if (editor) editor.setValue(s); };

require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" } });
require(["vs/editor/editor.main"], () => {
  editor = monaco.editor.create(document.getElementById("editor"), {
    value: initialSource,
    language: "python",
    theme: "vs-dark",
    fontSize: 13,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    automaticLayout: true,
    tabSize: 4,
  });
});

zoo.onchange = () => setSource(KERNELS[+zoo.value].source);

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function checkHealth() {
  const g = $("#gpu");
  try {
    await api("/api/health");
    g.textContent = "● live on Apple Silicon";
    g.className = "gpu up";
  } catch {
    g.textContent = "● API unreachable";
    g.className = "gpu down";
  }
}

async function poll(jid) {
  for (;;) {
    const r = await api(`/api/job/${jid}`);
    if (r.status === "done") return r.result;
    await new Promise(x => setTimeout(x, 300));
  }
}

const esc = s => (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function render(res) {
  const out = $("#out"); out.innerHTML = "";
  const banner = document.createElement("div");
  const okRun = res.ok && res.err != null;
  banner.className = "banner " + (okRun ? "ok" : "err");
  banner.textContent = okRun
    ? `✓ compiled through the AppleGPU backend + ran on the GPU, max|err| = ${res.err}`
    : "✗ " + (res.error || "failed");
  out.appendChild(banner);

  if (res.pso && res.pso.ok) {
    const p = res.pso, tg = p.staticThreadgroupMemoryLength;
    const bar = document.createElement("div"); bar.className = "pso";
    bar.innerHTML =
      `<span><b>${p.maxTotalThreadsPerThreadgroup}</b> max threads/threadgroup</span>` +
      `<span><b>${p.threadExecutionWidth}</b> SIMD width</span>` +
      `<span><b>${tg}</b> B static TG${tg === 0 ? " (0 = dynamic/none)" : ""}</span>`;
    out.appendChild(bar);
  }

  const toggle = div => {
    const wasOpen = div.classList.contains("open");
    out.querySelectorAll(".stage.open").forEach(s => s.classList.remove("open"));
    if (!wasOpen) div.classList.add("open");
  };
  let openTarget = null;
  for (const s of res.stages || []) {
    const div = document.createElement("div"); div.className = "stage";
    const cls = s.ok ? "ok" : (res.error ? "err" : "skip");
    const body = s.artifact || "(not reached)";
    const isLib = s.name === "metallib" && s.ok && res.has_metallib;
    if (s.ok && s.name === "air-ir") openTarget = div;        // default expanded
    const dl = isLib
      ? `<a class="dl" href="${API}/api/bundle/${res.job_id}" download>⤓ Download standalone runner (.zip)</a>`
      : "";
    div.innerHTML = `<div class="stage-h"><span class="dot ${cls}"></span>
        <span class="name">${s.name}</span>
        <span class="meta">${esc(s.label || "")}</span></div>
      <div class="stage-body"><pre>${esc(body)}</pre>${dl}</div>`;
    div.querySelector(".stage-h").onclick = () => toggle(div);
    out.appendChild(div);
  }
  if (openTarget) openTarget.classList.add("open");

  if (res.log) {
    const div = document.createElement("div"); div.className = "stage";
    div.innerHTML = `<div class="stage-h"><span class="dot skip"></span>
        <span class="name">stderr</span></div>
      <div class="stage-body"><pre>${esc(res.log)}</pre></div>`;
    div.querySelector(".stage-h").onclick = () => toggle(div);
    out.appendChild(div);
  }
}

$("#go").onclick = async () => {
  const btn = $("#go"); btn.disabled = true; btn.textContent = "Running…";
  $("#out").innerHTML = `<div class="hint">compiling @triton.jit → AIR → dispatching to the GPU…</div>`;
  try {
    const sub = await api("/api/compile", {
      method:"POST", headers:{"content-type":"application/json"},
      body: JSON.stringify({ source: getSource() })
    });
    $("#qd").textContent = sub.queue_depth > 0 ? `queue: ${sub.queue_depth}` : "";
    render(await poll(sub.job_id));
  } catch (e) {
    $("#out").innerHTML = `<div class="banner err">✗ ${esc(""+e)}</div>`;
  } finally {
    $("#qd").textContent = ""; btn.disabled = false; btn.textContent = "Compile & Run";
  }
};

checkHealth();
