const state = {
  datasets: [],
  files: [],
  activeDataset: null,
  agent: {
    sessionId: null,
    activeJobId: null,
    pollTimer: null,
    activeScript: "yichao",
  },
};

const pipelineScripts = {
  yichao: {
    title: "Yichao pix2pix differentiation prediction",
    databaseHint: "analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"yichao_pix2pix","title":"Yichao fluorescence prediction dataset","objective":"Prepare paired brightfield and fluorescence instances for pix2pix training."}
STEP {"id":"inspect","block":"plan","title":"Inspect data","instruction":"Check Yichao 1/2/3/4/5/6 structure, channel mapping, and existing instance-pair database."}
ACTION {"type":"read","target":"references/Yichao"}
STEP {"id":"segment","block":"work","title":"Segment brightfield","instruction":"Run the multiscale Cellpose segmentation pipeline on brightfield channel c1 and save overlays/intermediates."}
ACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}
STEP {"id":"pair","block":"work","title":"Build pix2pix pairs","instruction":"Crop matched c1 brightfield and c0 fluorescence instances, then resize or pad to 256x256."}
ACTION {"type":"dataset","target":"analysis-outputs/yichao_pix2pix_256"}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Backfill edge padding flags, size quantiles, source image metadata, and resized-pair links into SQLite."}
ACTION {"type":"database","target":"analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite"}
STEP {"id":"review","block":"summary","title":"Review quality","instruction":"Report edge padding, instance size quantiles, debris filtering risks, and preview paths."}`,
  },
  zhengyu: {
    title: "../Zhengyu segmentation and metric pipeline",
    databaseHint: "../Zhengyu",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"zhengyu_deo_metrics","title":"Zhengyu DEO segmentation metrics","objective":"Reuse the production multiscale segmentation and metric extraction workflow from ../Zhengyu."}
STEP {"id":"read_method","block":"plan","title":"Read canonical method","instruction":"Open the segmentation handoff and PDF references before editing or running scripts."}
ACTION {"type":"read","target":"../Zhengyu/references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"../Zhengyu/references/deo_segmentation_metric_method_tex/main.pdf"}
STEP {"id":"segment","block":"work","title":"Run multiscale segmentation","instruction":"Use the production Cellpose recovery pipeline and save masks, instance overlays, and recovery intermediates."}
ACTION {"type":"script","target":"../Zhengyu"}
STEP {"id":"quantify","block":"work","title":"Compute DEO metrics","instruction":"Extract growth, fusion, compactness, and differentiation metrics into the maintained database."}
ACTION {"type":"database","target":"../Zhengyu"}
STEP {"id":"validate","block":"debug","title":"Validate against references","instruction":"Compare outputs with saved intermediate masks and metric catalogs, then report deviations."}
STEP {"id":"report","block":"summary","title":"Summarize run","instruction":"Write output paths, database status, and quality-control notes."}`,
  },
  compactness: {
    title: "../Compactness compactness analysis",
    databaseHint: "../Compactness",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"compactness_analysis","title":"Compactness image-analysis pipeline","objective":"Build and maintain a compactness-focused organoid image analysis workflow."}
STEP {"id":"inspect","block":"plan","title":"Inspect Compactness repo","instruction":"Find the image sources, current scripts, database outputs, and expected compactness definitions."}
ACTION {"type":"read","target":"../Compactness"}
STEP {"id":"segment","block":"work","title":"Segment organoids","instruction":"Run or adapt multiscale brightfield segmentation and save instance masks and overlays."}
STEP {"id":"quantify","block":"work","title":"Quantify compactness","instruction":"Measure area, perimeter, solidity, eccentricity, texture, and compactness scores per instance."}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Store per-image and per-instance metrics with source paths, crop geometry, and QC flags."}
ACTION {"type":"database","target":"../Compactness"}
STEP {"id":"review","block":"debug","title":"Review edge cases","instruction":"Sample high/low compactness instances and flag debris, edge clipping, and bad focus."}
STEP {"id":"report","block":"summary","title":"Report dataset readiness","instruction":"Summarize usable images, metric distributions, and recommended filtering."}`,
  },
  generic: {
    title: "Generic organoid agent workflow",
    databaseHint: "analysis-outputs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"organoid_generic","title":"Generic organoid analysis task","objective":"Inspect data, segment objects, quantify instances, maintain a database, and report results."}
STEP {"id":"inspect","block":"plan","title":"Inspect task","instruction":"Identify inputs, channels, imaging design, output folders, and existing references."}
STEP {"id":"segment","block":"work","title":"Segment images","instruction":"Run the appropriate segmentation pipeline and save masks, overlays, and intermediates."}
STEP {"id":"quantify","block":"work","title":"Quantify instances","instruction":"Extract per-image and per-instance metrics with QC flags."}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Create or update the SQLite database and summary manifests."}
STEP {"id":"report","block":"summary","title":"Report outputs","instruction":"Document output paths, counts, filters, and next actions."}`,
  },
};

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
  return String(value ?? "")
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

function populateScriptSelect() {
  const select = document.getElementById("script-select");
  if (!select) {
    return;
  }
  select.innerHTML = Object.entries(pipelineScripts)
    .map(([key, script]) => `<option value="${escapeHtml(key)}">${escapeHtml(script.title)}</option>`)
    .join("");
  select.value = state.agent.activeScript;
}

function loadSelectedScript(forcePreset = false) {
  const select = document.getElementById("script-select");
  const editor = document.getElementById("pipeline-editor");
  if (!select || !editor) {
    return;
  }
  const key = select.value || "yichao";
  const localKey = `organoid-agent-script-${key}`;
  state.agent.activeScript = key;
  editor.value = !forcePreset && localStorage.getItem(localKey) ? localStorage.getItem(localKey) : pipelineScripts[key].text;
  parsePipeline();
}

function saveSelectedScript() {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const key = state.agent.activeScript || "yichao";
  localStorage.setItem(`organoid-agent-script-${key}`, editor.value);
  const status = document.getElementById("pipeline-status");
  if (status) {
    status.textContent = `Saved local edits for ${pipelineScripts[key].title}.`;
  }
}

function renderDatabases(databases) {
  const container = document.getElementById("database-registry");
  if (!container) {
    return;
  }
  if (!databases.length) {
    container.innerHTML = "<div class='muted'>No SQLite databases found.</div>";
    return;
  }
  container.innerHTML = databases
    .map((db) => {
      const tableRows = (db.tables || [])
        .map((table) => `<span class="db-table">${escapeHtml(table.table)}: ${escapeHtml(table.rows ?? "?")}</span>`)
        .join("");
      const summary = db.summary?.path ? `<div class="meta">summary: ${escapeHtml(db.summary.path)}</div>` : "";
      const error = db.error ? `<div class="meta danger">sqlite read error: ${escapeHtml(db.error)}</div>` : "";
      return `
        <div class="database-card">
          <div class="db-title">${escapeHtml(db.project)} · ${escapeHtml(db.name)}</div>
          <div class="meta">${escapeHtml(db.path)} · ${escapeHtml(db.size_human)}</div>
          <div class="db-tables">${tableRows || "<span class='db-table'>no tables read</span>"}</div>
          ${summary}
          ${error}
        </div>
      `;
    })
    .join("");
}

async function loadDatabaseRegistry() {
  const container = document.getElementById("database-registry");
  if (container) {
    container.textContent = "Loading databases...";
  }
  try {
    const data = await fetchJson("/api/agent/databases");
    renderDatabases(data.databases || []);
  } catch (err) {
    if (container) {
      container.textContent = `Database scan failed: ${err.message}`;
    }
  }
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
    const script = pipelineScripts[state.agent.activeScript] || pipelineScripts.generic;
    status.textContent = `Parsed ${data.ir.tasks.length} task(s), ${stepCount} step(s). Database hint: ${script.databaseHint}.`;
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
    inspect: 'STEP {"id":"inspect_next","block":"plan","title":"Inspect inputs","instruction":"Identify source folders, channels, time/depth design, references, scripts, and output requirements."}\nACTION {"type":"read","target":"references"}',
    segmentation: 'STEP {"id":"segment_next","block":"work","title":"Segment brightfield","instruction":"Run brightfield segmentation and save instance masks, overlays, and crops."}\nACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}',
    tracking: 'STEP {"id":"tracking_next","block":"work","title":"Track positions over time","instruction":"Link objects across day/time/position folders and record monitoring metadata."}\nACTION {"type":"database","target":"analysis-outputs"}',
    pairing: 'STEP {"id":"pair_next","block":"work","title":"Create paired crops","instruction":"Match brightfield c1 with fluorescence c0 and write paired dataset records."}\nACTION {"type":"dataset","target":"analysis-outputs/yichao_instance_pairs"}',
    quantification: 'STEP {"id":"quantify_next","block":"work","title":"Quantify instances","instruction":"Measure instance size, padding, edge contact, fluorescence intensity, and debris flags."}',
    database: 'STEP {"id":"database_next","block":"work","title":"Maintain database","instruction":"Create or update SQLite tables, summary manifests, schema notes, source paths, and QC filter fields."}\nACTION {"type":"database","target":"analysis-outputs"}',
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
  populateScriptSelect();
  loadSelectedScript();
  parsePipeline();
  loadAgentState();
  loadDatabaseRegistry();
  document.getElementById("script-select")?.addEventListener("change", () => loadSelectedScript());
  document.getElementById("load-script-btn")?.addEventListener("click", () => loadSelectedScript(true));
  document.getElementById("save-script-btn")?.addEventListener("click", saveSelectedScript);
  document.getElementById("parse-pipeline-btn")?.addEventListener("click", parsePipeline);
  document.getElementById("refresh-agent-state")?.addEventListener("click", loadAgentState);
  document.getElementById("refresh-database-registry")?.addEventListener("click", loadDatabaseRegistry);
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
