const state = {
  datasets: [],
  files: [],
  activeDataset: null,
  agent: {
    sessionId: null,
    activeJobId: null,
    pollTimer: null,
  },
};

const defaultPipeline = `AUTOAPPDEV_PIPELINE 1
TASK {"id":"yichao_pix2pix","title":"Yichao fluorescence prediction dataset","objective":"Prepare paired brightfield and fluorescence instances for pix2pix training."}
STEP {"id":"inspect","block":"plan","title":"Inspect data","instruction":"Check Yichao 1/2/3/4/5/6 structure, channel mapping, and existing instance-pair database."}
ACTION {"type":"read","target":"references/Yichao"}
STEP {"id":"segment","block":"work","title":"Segment brightfield","instruction":"Run the multiscale Cellpose segmentation pipeline on brightfield channel c1 and save overlays/intermediates."}
ACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}
STEP {"id":"pair","block":"work","title":"Build pix2pix pairs","instruction":"Crop matched c1 brightfield and c0 fluorescence instances, then resize or pad to 256x256."}
ACTION {"type":"dataset","target":"analysis-outputs/yichao_pix2pix_256"}
STEP {"id":"review","block":"summary","title":"Review quality","instruction":"Report edge padding, instance size quantiles, debris filtering risks, and preview paths."}`;

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
}

async function postJson(url, payload) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function setActiveTab(tabName) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

function renderList(containerId, items, onClick) {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = "<div class='muted'>No items found.</div>";
    return;
  }
  container.innerHTML = "";
  items.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "list-item";
    div.style.animationDelay = `${idx * 0.02}s`;
    div.innerHTML = `
      <div><strong>${item.name || item.path}</strong></div>
      <div class="meta">${item.size_human || ""} ${item.kind ? `• ${item.kind}` : ""}</div>
    `;
    div.addEventListener("click", () => onClick(item));
    container.appendChild(div);
  });
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(text) {
  const pattern = /\[([^\]]+)\]\(([^)]+)\)/g;
  let result = "";
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    result += escapeHtml(text.slice(lastIndex, match.index));
    const label = escapeHtml(match[1]);
    const url = match[2];
    if (url.startsWith("http://") || url.startsWith("https://")) {
      result += `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    } else {
      result += `${label} (${escapeHtml(url)})`;
    }
    lastIndex = match.index + match[0].length;
  }
  result += escapeHtml(text.slice(lastIndex));
  return result;
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let inList = false;
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      return;
    }
    if (trimmed.startsWith("### ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h4>${renderInlineMarkdown(trimmed.slice(4))}</h4>`;
      return;
    }
    if (trimmed.startsWith("## ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h3>${renderInlineMarkdown(trimmed.slice(3))}</h3>`;
      return;
    }
    if (trimmed.startsWith("# ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h3>${renderInlineMarkdown(trimmed.slice(2))}</h3>`;
      return;
    }
    if (trimmed.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${renderInlineMarkdown(trimmed.slice(2))}</li>`;
      return;
    }
    if (inList) {
      html += "</ul>";
      inList = false;
    }
    html += `<p>${renderInlineMarkdown(trimmed)}</p>`;
  });
  if (inList) {
    html += "</ul>";
  }
  return html;
}

function renderPreview(containerId, payload) {
  const container = document.getElementById(containerId);
  if (!payload) {
    container.textContent = "No preview.";
    return;
  }

  if (payload.error) {
    container.textContent = payload.error;
    return;
  }

  if (payload.kind === "table" && payload.preview?.columns) {
    const rows = payload.preview.rows.slice(0, 15);
    const headers = payload.preview.columns;
    const html = [
      "<table><thead><tr>",
      ...headers.map((h) => `<th>${h}</th>`),
      "</tr></thead><tbody>",
      ...rows.map(
        (row) =>
          `<tr>${row.map((v) => `<td>${String(v)}</td>`).join("")}</tr>`
      ),
      "</tbody></table>",
    ].join("");
    container.innerHTML = html;
    return;
  }

  if (payload.kind === "analysis" && payload.preview) {
    const summary = payload.preview;
    const imageHtml = summary.preview_url
      ? `<img src="${summary.preview_url}" alt="${payload.name}" />`
      : "";
    const previewMeta = summary.preview_url
      ? `<p>Embedding: ${summary.preview_source} (${summary.preview_points} points)</p>`
      : summary.preview_error
      ? `<p class="muted">Preview: ${summary.preview_error}</p>`
      : "";
    container.innerHTML = `
      <div class="tag">AnnData</div>
      ${imageHtml}
      ${previewMeta}
      <p>Observations: ${summary.n_obs}</p>
      <p>Variables: ${summary.n_vars}</p>
      <p><strong>Obs columns:</strong> ${summary.obs_columns.join(", ") || "—"}</p>
      <p><strong>Var columns:</strong> ${summary.var_columns.join(", ") || "—"}</p>
      <p><strong>Uns keys:</strong> ${summary.uns_keys.join(", ") || "—"}</p>
    `;
    return;
  }

  if (payload.kind === "image" && payload.preview?.preview_url) {
    container.innerHTML = `<img src="${payload.preview.preview_url}" alt="${payload.name}" />`;
    return;
  }

  if (payload.kind === "archive" && payload.preview?.entries) {
    const entries = payload.preview.entries
      .map((entry) => `<div>${entry.name}</div>`)
      .join("");
    const previewImage = payload.preview.preview_url
      ? `<div class="muted">Preview: ${payload.preview.preview_entry}</div><img src="${payload.preview.preview_url}" alt="Archive preview" />`
      : "";
    container.innerHTML = `
      <div class="tag">Archive</div>
      <button class="tab" id="extract-btn">Extract</button>
      ${previewImage}
      <div class="preview-body" style="margin-top:10px">${entries || "No entries."}</div>
    `;
    const btn = container.querySelector("#extract-btn");
    if (btn) {
      btn.addEventListener("click", async () => {
        btn.textContent = "Extracting…";
        try {
          const res = await fetchJson(`/api/extract?path=${payload.path}`, {
            method: "POST",
          });
          btn.textContent = `Extracted: ${res.extracted_to}`;
        } catch (err) {
          btn.textContent = "Extract failed";
        }
      });
    }
    return;
  }

  if (payload.preview?.download_url) {
    container.innerHTML = `<a href="${payload.preview.download_url}" target="_blank">Download ${payload.name}</a>`;
    return;
  }

  if (payload.preview?.lines) {
    container.innerHTML = `<pre>${payload.preview.lines.join("\n")}</pre>`;
    return;
  }

  container.textContent = "Preview not available.";
}

function appendAgentMessage(role, content) {
  const container = document.getElementById("agent-messages");
  if (!container) {
    return;
  }
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.innerHTML = `<div class="chat-role">${escapeHtml(role)}</div><div>${escapeHtml(content)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function renderPipelineBlocks(ir) {
  const canvas = document.getElementById("pipeline-blocks");
  if (!canvas) {
    return;
  }
  canvas.innerHTML = "";
  ir.tasks.forEach((task) => {
    const taskEl = document.createElement("div");
    taskEl.className = "program-block task";
    taskEl.innerHTML = `<strong>${escapeHtml(task.title)}</strong><span>${escapeHtml(task.objective || task.id)}</span>`;
    canvas.appendChild(taskEl);
    task.steps.forEach((step) => {
      const stepEl = document.createElement("div");
      stepEl.className = `program-block ${step.block}`;
      stepEl.innerHTML = `
        <strong>${escapeHtml(step.title)}</strong>
        <span>${escapeHtml(step.instruction || step.id)}</span>
        <small>${escapeHtml(step.block)} • ${step.actions.length} actions</small>
      `;
      canvas.appendChild(stepEl);
    });
  });
}

async function parsePipeline() {
  const editor = document.getElementById("pipeline-editor");
  const status = document.getElementById("pipeline-status");
  if (!editor || !status) {
    return null;
  }
  status.textContent = "Parsing pipeline...";
  try {
    const data = await postJson("/api/agent/pipeline/parse", { text: editor.value });
    renderPipelineBlocks(data.ir);
    const stepCount = data.ir.tasks.reduce((sum, task) => sum + task.steps.length, 0);
    status.textContent = `Parsed ${data.ir.tasks.length} task(s), ${stepCount} step(s).`;
    return data.ir;
  } catch (err) {
    status.textContent = `Parse failed: ${err.message}`;
    return null;
  }
}

async function loadAgentState() {
  const grid = document.getElementById("agent-status-grid");
  if (!grid) {
    return;
  }
  try {
    const data = await fetchJson("/api/agent/state");
    grid.innerHTML = `
      <div class="status-row"><span>Backend</span><strong>ok</strong></div>
      <div class="status-row"><span>Codex</span><strong>${data.codex_available ? "available" : "missing"}</strong></div>
      <div class="status-row"><span>Model</span><strong>${escapeHtml(data.default_model)}</strong></div>
      <div class="status-row"><span>Jobs</span><strong>${data.recent_jobs.length}</strong></div>
    `;
  } catch (err) {
    grid.innerHTML = `<div class="status-row"><span>Backend</span><strong>error</strong></div>`;
  }
}

async function ensureAgentSession() {
  if (state.agent.sessionId) {
    return state.agent.sessionId;
  }
  const data = await postJson("/api/agent/session", { title: "OrganoidAgent Studio" });
  state.agent.sessionId = data.session.id;
  return state.agent.sessionId;
}

function setAgentOutput(text) {
  const output = document.getElementById("agent-job-output");
  if (output) {
    output.textContent = text || "No output yet.";
  }
}

async function pollAgentJob(jobId) {
  if (!jobId) {
    return;
  }
  try {
    const data = await fetchJson(`/api/agent/codex/job?id=${encodeURIComponent(jobId)}`);
    const job = data.job;
    const body = data.output_text || data.logs?.stderr_tail || data.logs?.stdout_tail || "Waiting for Codex output...";
    setAgentOutput(`[${job.status}] ${job.id}\n\n${body}`);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      clearInterval(state.agent.pollTimer);
      state.agent.pollTimer = null;
      appendAgentMessage("assistant", data.output_text || `Job ${job.status}: ${job.id}`);
    }
  } catch (err) {
    setAgentOutput(`Poll failed: ${err.message}`);
  }
}

async function submitAgentJob(tool, prompt) {
  const pipelineText = document.getElementById("pipeline-editor")?.value || "";
  const sessionId = await ensureAgentSession();
  const data = await postJson("/api/agent/chat", {
    session_id: sessionId,
    message: prompt,
    tool,
    pipeline_text: pipelineText,
    allow_edits: tool === "assistant",
  });
  state.agent.activeJobId = data.job.id;
  appendAgentMessage("user", prompt);
  appendAgentMessage("assistant", `Started ${tool} job ${data.job.id}.`);
  setAgentOutput(`[queued] ${data.job.id}`);
  if (state.agent.pollTimer) {
    clearInterval(state.agent.pollTimer);
  }
  state.agent.pollTimer = setInterval(() => pollAgentJob(data.job.id), 2500);
  pollAgentJob(data.job.id);
}

function insertPipelineTemplate(kind) {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const snippets = {
    segmentation: 'STEP {"id":"segment_next","block":"work","title":"Segment brightfield","instruction":"Run brightfield segmentation and save instance masks, overlays, and crops."}\nACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}',
    pairing: 'STEP {"id":"pair_next","block":"work","title":"Create paired crops","instruction":"Match brightfield c1 with fluorescence c0 and write paired dataset records."}\nACTION {"type":"dataset","target":"analysis-outputs/yichao_instance_pairs"}',
    quantification: 'STEP {"id":"quantify_next","block":"work","title":"Quantify instances","instruction":"Measure instance size, padding, edge contact, fluorescence intensity, and debris flags."}',
    review: 'STEP {"id":"review_next","block":"debug","title":"Human review gate","instruction":"Sample random pairs and identify debris, edge-padded crops, and wrong channel assignments."}',
    report: 'STEP {"id":"report_next","block":"summary","title":"Write report","instruction":"Summarize outputs, database paths, histograms, and training recommendations."}',
  };
  editor.value = `${editor.value.trim()}\n${snippets[kind] || ""}\n`;
  parsePipeline();
}

function initAgentStudio() {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  editor.value = defaultPipeline;
  parsePipeline();
  loadAgentState();
  document.getElementById("parse-pipeline-btn")?.addEventListener("click", parsePipeline);
  document.getElementById("refresh-agent-state")?.addEventListener("click", loadAgentState);
  document.getElementById("new-agent-chat")?.addEventListener("click", async () => {
    const data = await postJson("/api/agent/session", { title: "OrganoidAgent Studio" });
    state.agent.sessionId = data.session.id;
    document.getElementById("agent-messages").innerHTML = "";
    setAgentOutput(`New session ${data.session.id}`);
  });
  document.getElementById("send-plan-btn")?.addEventListener("click", () => {
    submitAgentJob("response", "Review this AAPS pipeline and suggest the next concrete OrganoidAgent implementation step.");
  });
  document.getElementById("run-assistant-btn")?.addEventListener("click", () => {
    submitAgentJob("assistant", "Use this AAPS pipeline as the current plan and make the next safe implementation change in this repository.");
  });
  document.getElementById("send-agent-chat")?.addEventListener("click", () => {
    const input = document.getElementById("agent-chat-input");
    const message = input.value.trim();
    if (!message) {
      return;
    }
    const tool = document.getElementById("assistant-mode-toggle")?.checked ? "assistant" : "response";
    input.value = "";
    submitAgentJob(tool, message);
  });
  document.querySelectorAll("[data-template]").forEach((button) => {
    button.addEventListener("click", () => insertPipelineTemplate(button.dataset.template));
  });
}

async function loadDatasetMetadata(dataset) {
  const container = document.getElementById("dataset-info");
  if (!container) {
    return;
  }
  container.textContent = "Loading metadata…";
  try {
    const data = await fetchJson(`/api/datasets/${dataset}/metadata`);
    if (!data.markdown) {
      container.textContent = "No metadata available.";
      return;
    }
    container.innerHTML = renderMarkdown(data.markdown);
  } catch (err) {
    container.textContent = "No metadata available.";
  }
}

async function loadDatasets() {
  const data = await fetchJson("/api/datasets");
  state.datasets = data.datasets;
  const totalSize = data.datasets
    .reduce((acc, ds) => acc + ds.size_bytes, 0);
  document.getElementById(
    "dataset-stats"
  ).textContent = `${data.datasets.length} datasets • ${(
    totalSize /
    (1024 * 1024 * 1024)
  ).toFixed(2)} GB`;

  renderList("dataset-list", data.datasets, (item) => {
    state.activeDataset = item.path;
    loadDatasetFiles(item.path);
  });

  if (data.datasets.length) {
    state.activeDataset = data.datasets[0].path;
    loadDatasetFiles(state.activeDataset);
  }
}

async function loadDatasetFiles(dataset) {
  const data = await fetchJson(`/api/datasets/${dataset}`);
  state.files = data.files;
  renderList("file-list", data.files, async (file) => {
    const preview = await fetchJson(`/api/preview?path=${file.path}`);
    renderPreview("preview-panel", preview);
  });
  loadDatasetMetadata(dataset);
}

async function loadCategory(category, listId, previewId) {
  const data = await fetchJson(`/api/category/${category}`);
  renderList(listId, data.files, async (file) => {
    const preview = await fetchJson(`/api/preview?path=${file.path}`);
    renderPreview(previewId, preview);
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveTab(tab.dataset.tab);
  });
});

loadDatasets().catch((err) => {
  document.getElementById("dataset-stats").textContent = err.message;
});

loadCategory("segmentation", "segmentation-list", "preview-panel").catch(() => {});
loadCategory("features", "features-list", "features-preview").catch(() => {});
loadCategory("analysis", "analysis-list", "analysis-preview").catch(() => {});
initAgentStudio();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/static/sw.js", { scope: "/" });
}
