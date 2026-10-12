const PIPELINE_PRESET_VERSION = "14";

const state = {
  datasets: [],
  files: [],
  activeDataset: null,
  fluorescence: {
    images: [],
    selected: new Set(),
    running: false,
    progressTimer: null,
  },
  results: {
    records: [],
    selectedId: null,
    recordedJobIds: new Set(),
  },
  agent: {
    sessionId: null,
    activeJobId: null,
    morphologyJobId: null,
    morphologyStatus: null,
    morphologyInstruction: "",
    morphologyAnalysisJobId: null,
    morphologyResultRecordId: null,
    viabilityResultRecordId: null,
    pollTimer: null,
    activeScript: "organoid_analysis",
    selectedDataset: "",
    availableFiles: [],
    selectedFiles: new Set(),
    currentTaskStartedAt: null,
  },
};

const pipelineScripts = {
  neo_cpsam: {
    title: "NEO CellposeSAM morphology analysis",
    databaseHint: "analysis-outputs/morphology_jobs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"neo_cpsam","title":"NEO CellposeSAM morphology analysis","objective":"Segment selected NEO images with Cellpose 4.0.5 pretrained cpsam using default inference parameters and calculate per-instance morphology."}
STEP {"id":"inputs","block":"plan","title":"Resolve selected NEO images","instruction":"Use only selected JPG images from 07_NEO."}
ACTION {"type":"dataset","target":"selected_files"}
STEP {"id":"method","block":"plan","title":"Read morphology definitions","instruction":"Use the repository NEO workflow reference for metric definitions, units and QC rules."}
ACTION {"type":"read","target":"references/neo_cpsam_morphology_workflow.md"}
STEP {"id":"segment","block":"work","title":"Run CellposeSAM","instruction":"POST one request to /api/morphology/run with workflow_id neo_cpsam and the selected relative paths. The managed runner uses Cellpose 4.0.5 cpsam and does not override eval parameters."}
ACTION {"type":"api","target":"/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Calculate morphology","instruction":"Export OrganoID-compatible area, perimeter, circularity, solidity and eccentricity plus OrganoSeg2-style axes, equivalent/Feret diameters, extent, centroid, orientation, radius, intensity and border flags."}
STEP {"id":"report","block":"summary","title":"Generate the summary table and interpretation","instruction":"Use only the runner-produced report and results table. Present image, instance count, measured total projected area, and mean projected area in a Markdown table. Report px² unless spatial calibration is explicitly supplied. Mention segmentation QC and do not infer volume, time-course growth, physical units, or biological conclusions."}`,
  },
  organoid_profiler: {
    title: "24-Dimension Organoid Profiling",
    databaseHint: "analysis-outputs/organoid_profiler",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"organoid_profiler","title":"24-Dimension Organoid Profiling","objective":"Use YOLO plus SAM on the selected 49 longitudinal organoid TIFF groups, then calculate the repository-local 24-dimensional morphology profile."}
STEP {"id":"inputs","block":"plan","title":"Resolve selected TIFFs and scale","instruction":"Use only selected files from 06_OrganoidProfiler_49. If pixel size is blank, report image-space units. If supplied, pass the positive pixel_size value and report physical length/area units."}
ACTION {"type":"dataset","target":"selected_files"}
STEP {"id":"segment","block":"work","title":"Run YOLO plus SAM and profile","instruction":"POST exactly one request to http://127.0.0.1:8080/api/organoid-profiler/run with selected relative_path files, optional pixel_size, and generate_charts=true when the user requests a graph. Use YOLO then SAM only; do not use Cellpose or the viability checkpoint. Preserve masks, overlays, crops, CSV, JSON and chart files."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/organoid-profiler/run"}
STEP {"id":"charts","block":"work","title":"Generate requested statistics figures","instruction":"If a graph is requested, generate longitudinal area and growth-rate line charts with individual group trajectories and mean ± standard deviation. If another workflow produces a results CSV or JSON, use POST http://127.0.0.1:8080/api/charts/generate instead of inventing values."}
ACTION {"type":"chart","target":"results_table"}
STEP {"id":"report","block":"summary","title":"Report 24 dimensions and longitudinal growth","instruction":"Report the 24 parameters per image, the day/group mapping, Day 0 growth rate, units, segmentation confidence, and any failed or manually reviewable images."}`,
  },
  organoid_analysis: {
    title: "Viability Detection",
    databaseHint: "analysis-outputs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"organoid_analysis","title":"Viability Detection","objective":"Analyze the selected images in the shared Agent workflow, preserve segmentation and viability evidence, and generate only the result summaries, tables, and charts requested by the user."}
STEP {"id":"inspect","block":"plan","title":"Resolve selected inputs and requested analyses","instruction":"Use only the selected dataset files. Read the user's prompt and determine whether it requests brightfield morphology, viability/activity prediction, or both. Do not expand the input set."}
ACTION {"type":"dataset","target":"selected_files"}
STEP {"id":"route","block":"plan","title":"Select existing tools","instruction":"For morphology requests, use the existing Cellpose-first managed morphology endpoint with the appropriate workflow_id (zhengyu, y27632, or sodium_alginate). For viability requests, use the existing fluorescence endpoint, whose morphology stage is YOLO plus SAM followed by the existing viability model. Do not substitute algorithms."}
STEP {"id":"morphology","block":"work","title":"Run brightfield morphology when requested","instruction":"POST exactly one managed morphology request per requested morphology dataset/analysis to http://127.0.0.1:8080/api/morphology/run. Preserve the selected relative_path values and the existing Cellpose QC, signal-recovery, final-mask, metrics, and report behavior. Return the accepted job id without launching the runner from a shell."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"viability","block":"work","title":"Run viability when requested","instruction":"POST the selected relative_path image files to http://127.0.0.1:8080/api/fluorescence/run. This existing endpoint must perform YOLO detection plus SAM segmentation and the existing viability model, preserving its crop, mask, overlay, score, CSV, JSON, and report outputs. Do not replace it with Cellpose."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/fluorescence/run"}
STEP {"id":"charts","block":"work","title":"Generate requested result charts","instruction":"After the viability results CSV or JSON exists, inspect the user's instruction. When a table or chart is requested, POST the real result file and requested viability/morphology fields to http://127.0.0.1:8080/api/charts/generate. Never invent values or draw a chart before the analysis result exists."}
ACTION {"type":"chart","target":"http://127.0.0.1:8080/api/charts/generate"}
STEP {"id":"report","block":"summary","title":"AI result summary and evidence report","instruction":"This is a mandatory final workflow step after viability detection. Call the result-summary Agent exactly once using the completed results JSON/CSV; never skip it and never rerun segmentation. The Agent must produce a Markdown table sorted from highest to lowest predicted viability with image name, predicted viability, classification, valid detection-region count, and analysis status. It must summarize highest/lowest values and the batch trend, preserve failure reasons, and link the actual overlay, mask, crop, and other evidence files. If the user's request asks for a chart, call the real chart-generation endpoint and embed the generated chart artifact; never fabricate chart values. Persist the Agent summary into the canonical report.md. Do not infer biological causality from image predictions."}`,
  },
  zhengyu: {
    title: "App81 density segmentation and metric pipeline",
    databaseHint: "analysis-outputs/density_growth",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"zhengyu_deo_metrics","title":"App81 density segmentation and morphology metrics","objective":"Run the original App81 condition/day-aware multiscale Cellpose, signal-derived large-object recovery, overlap-aware candidate merge, and metric workflow."}
STEP {"id":"read_method","block":"plan","title":"Read canonical method","instruction":"Open the repository-local segmentation handoff and readable TeX method source before running scripts."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/deo_segmentation_metric_method_tex/main.tex"}
STEP {"id":"segment","block":"work","title":"Submit managed multiscale segmentation","instruction":"Do not run the Cellpose script in a Codex shell. POST one JSON request to http://127.0.0.1:8080/api/morphology/run using only each selected file's relative_path value in paths, workflow_id zhengyu, the user's exact instruction, and this Studio job id in parent_job_id. Never replace relative_path with absolute_path. The backend owns the long-running Cellpose process and derives the requested output scope from the instruction. Report the accepted morphology job id and return without waiting for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute DEO metrics","instruction":"Use the metrics produced by the same repository-local runner to summarize growth, fusion, compactness, and differentiation evidence for the selected images."}
ACTION {"type":"database","target":"analysis-outputs/density_growth"}
STEP {"id":"validate","block":"debug","title":"Validate against references","instruction":"Compare outputs with saved intermediate masks and metric catalogs, then report deviations."}
STEP {"id":"report","block":"summary","title":"Summarize run","instruction":"Write output paths, database status, and quality-control notes."}`,
  },
  y27632: {
    title: "Y-27632 dataset analysis",
    databaseHint: "analysis-outputs/morphology_jobs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"y27632_morphology","title":"Y-27632 dataset analysis","objective":"Analyze selected Y-27632 TIFFs using the original App80 date-aware multiscale Cellpose, signal-based large-mass recovery, candidate merge, morphology metrics, masks, overlays, and report."}
STEP {"id":"read_method","block":"plan","title":"Read App80 method","instruction":"Read the repository segmentation handoff and App80 Y-27632 method notes before submission."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/app80_multiscale_cellpose_large_recovery_method.md"}
STEP {"id":"segment","block":"work","title":"Submit managed Y-27632 analysis","instruction":"POST exactly one request to http://127.0.0.1:8080/api/morphology/run using only each selected TIFF's relative_path value in paths, workflow_id y27632, the user's exact instruction, and this Studio job id in parent_job_id. Never replace relative_path with absolute_path. The backend derives the requested output scope from the instruction. Do not launch Cellpose from a Codex shell and do not wait for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute concentration-aware metrics","instruction":"Use the final original App80 merged instance mask per image to quantify growth, fusion-context, roundness, edge, darkness, and concentration/day comparisons."}
STEP {"id":"report","block":"summary","title":"Report Y-27632 results","instruction":"Report the managed job id, output directory, QC method, and concentration/day analysis artifacts."}`,
  },
  sodium_alginate: {
    title: "Sodium alginate dataset analysis",
    databaseHint: "analysis-outputs/morphology_jobs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"sodium_alginate_morphology","title":"Sodium alginate dataset analysis","objective":"Analyze selected sodium alginate TIFFs using the original App65 condition/day-aware multiscale Cellpose, signal-based large-mass recovery, candidate merge, morphology metrics, masks, overlays, and report."}
STEP {"id":"read_method","block":"plan","title":"Read App65 method","instruction":"Read the repository segmentation handoff and sodium alginate experiment-design notes before submission."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/experiment_design/06_figure3_alginate_design.md"}
STEP {"id":"segment","block":"work","title":"Submit managed sodium alginate analysis","instruction":"POST one JSON request to http://127.0.0.1:8080/api/morphology/run using only each selected TIFF's relative_path value in paths, workflow_id sodium_alginate, the user's exact instruction, and this Studio job id in parent_job_id. Never replace relative_path with absolute_path. The backend owns the long-running Cellpose process and derives the requested output scope from the instruction. Do not launch Cellpose from a Codex shell and do not wait for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute condition-aware metrics","instruction":"Use the final original App65 merged instance mask per image to quantify growth, fusion-context, differentiation morphology, edge, darkness, and condition/day comparisons."}
STEP {"id":"report","block":"summary","title":"Report sodium alginate results","instruction":"Report the managed job id, output directory, QC method, and condition/day analysis artifacts."}`,
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
  document.querySelectorAll(".subtab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

const stageDefaults = {
  data: "datasets",
  workflow: "agent",
  results: "viability-results",
};

function setActiveStage(stageName, targetTab = null) {
  document.querySelectorAll(".stage-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.stage === stageName);
  });
  document.querySelectorAll(".subtab-group").forEach((group) => {
    group.classList.toggle("active", group.dataset.stageGroup === stageName);
  });
  setActiveTab(targetTab || stageDefaults[stageName]);
}

function renderList(containerId, items, onClick, emptyMessage = "No items found.") {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = `<div class='muted'>${escapeHtml(emptyMessage)}</div>`;
    return;
  }
  container.innerHTML = "";
  items.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "list-item";
    div.style.animationDelay = `${idx * 0.02}s`;
    const ownership = [item.dataset, item.group].filter(Boolean).map(escapeHtml).join(" · ");
    div.innerHTML = `
      <div><strong>${escapeHtml(item.name || item.path)}</strong></div>
      ${ownership ? `<div class="meta"><strong>${ownership}</strong></div>` : ""}
      <div class="meta">${escapeHtml(item.path || "")}</div>
      <div class="meta">${escapeHtml(item.size_human || "")} ${item.kind ? `• ${escapeHtml(item.kind)}` : ""}</div>
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
  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };
  const tableCells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  for (let index = 0; index < lines.length; index += 1) {
    const trimmed = lines[index].trim();
    if (!trimmed) {
      closeList();
      continue;
    }
    const next = (lines[index + 1] || "").trim();
    if (trimmed.startsWith("|") && /^\|?\s*:?-{3,}/.test(next)) {
      closeList();
      const headers = tableCells(trimmed);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      index -= 1;
      html += `<div class="markdown-table-wrap"><table><thead><tr>${headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      continue;
    }
    if (trimmed.startsWith("### ")) {
      closeList();
      html += `<h4>${renderInlineMarkdown(trimmed.slice(4))}</h4>`;
      continue;
    }
    if (trimmed.startsWith("## ")) {
      closeList();
      html += `<h3>${renderInlineMarkdown(trimmed.slice(3))}</h3>`;
      continue;
    }
    if (trimmed.startsWith("# ")) {
      closeList();
      html += `<h3>${renderInlineMarkdown(trimmed.slice(2))}</h3>`;
      continue;
    }
    if (trimmed.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${renderInlineMarkdown(trimmed.slice(2))}</li>`;
      continue;
    }
    closeList();
    html += `<p>${renderInlineMarkdown(trimmed)}</p>`;
  }
  closeList();
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
  const key = select.value || "organoid_analysis";
  const localKey = `organoid-agent-script-${key}`;
  const versionKey = `${localKey}-preset-version`;
  state.agent.activeScript = key;
  const savedScript = localStorage.getItem(localKey);
  const savedVersion = localStorage.getItem(versionKey);
  editor.value = !forcePreset && savedScript && savedVersion === PIPELINE_PRESET_VERSION ? savedScript : pipelineScripts[key].text;
  if (savedVersion !== PIPELINE_PRESET_VERSION) {
    localStorage.removeItem(localKey);
    localStorage.setItem(versionKey, PIPELINE_PRESET_VERSION);
  }
  parsePipeline();
}

function saveSelectedScript() {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const key = state.agent.activeScript || "organoid_analysis";
  localStorage.setItem(`organoid-agent-script-${key}`, editor.value);
  localStorage.setItem(`organoid-agent-script-${key}-preset-version`, PIPELINE_PRESET_VERSION);
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
    const script = pipelineScripts[state.agent.activeScript] || pipelineScripts.organoid_analysis;
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
    let fluorescenceStatus = null;
    try {
      fluorescenceStatus = await fetchJson("/api/fluorescence/status");
      resumeFluorescenceProgress(fluorescenceStatus);
    } catch (_) {
      // Agent status remains usable if the optional viability status request fails.
    }
    const activeMorphology = (data.active_morphology_jobs || [])[0];
    const activeAgent = (data.active_jobs || [])[0];
    if (activeMorphology) {
      state.agent.morphologyJobId = activeMorphology.id || activeMorphology.job_id;
      state.agent.morphologyStatus = activeMorphology.status;
      setStopJobEnabled(true);
      updateMorphologyProgressView(activeMorphology.progress || {}, activeMorphology.status);
      setAgentOutput(`[${activeMorphology.status}] Managed morphology job ${state.agent.morphologyJobId}\n\n${morphologyProgressText(activeMorphology.progress || {}, activeMorphology.status)}`);
    } else if (activeAgent) {
      state.agent.activeJobId = activeAgent.id;
      state.agent.currentTaskStartedAt = state.agent.currentTaskStartedAt || Date.now();
      updateCurrentTaskProgress("Current Analysis Progress", `Agent task status: ${activeAgent.status || "running"}.`, null, activeAgent.id, true);
      setStopJobEnabled(true);
      if (!state.agent.pollTimer) {
        state.agent.pollTimer = setInterval(() => pollAgentJob(activeAgent.id), 2500);
        pollAgentJob(activeAgent.id);
      }
    }
    const localMorphologyActive = state.agent.morphologyJobId && ["queued", "running", "pending"].includes(state.agent.morphologyStatus);
    const activeJobCount = Math.max(
      Number(data.active_job_count ?? (data.active_jobs || []).length),
      localMorphologyActive ? 1 : 0,
      fluorescenceStatus?.busy ? 1 : 0,
    );
    if (!activeJobCount) {
      updateCurrentTaskProgress("Current Task Progress", "No task running.", null, "", false);
    }
    grid.innerHTML = `
      <div class="status-row"><span>Backend</span><strong>ok</strong></div>
      <div class="status-row"><span>Codex</span><strong>${data.codex_available ? "available" : "missing"}</strong></div>
      <div class="status-row"><span>Model</span><strong>${escapeHtml(data.default_model)}</strong></div>
      <div class="status-row"><span>Active Jobs</span><strong>${activeJobCount}</strong></div>
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

function setStopJobEnabled(enabled) {
  const button = document.getElementById("stop-agent-job");
  if (button) button.disabled = !enabled;
}

function morphologyProgressText(progress = {}, status = "pending") {
  const percent = Number(progress.percent || 0);
  const phase = progress.phase ? ` Phase: ${progress.phase}.` : "";
  const current = progress.current_file ? ` Current: ${progress.current_file}.` : "";
  const scale = progress.current_scale ? ` Cellpose scale ${progress.scale_index}/${progress.scale_total}: ${progress.current_scale}px.` : "";
  const inference = status === "running" && progress.phase === "cellpose_cpsam" ? " GPU inference is in progress; this counter advances after the current image finishes." : "";
  return `${progress.completed || 0}/${progress.total || 0} completed (${percent.toFixed(1)}%).${phase}${current}${scale} Elapsed: ${Number(progress.elapsed_seconds || 0).toFixed(0)} s. Status: ${status}.${inference}`;
}

function resumeFluorescenceProgress(status = {}) {
  const current = status.progress;
  const active = Boolean(status.busy && status.active_job_id && current);
  if (!active) return false;
  state.fluorescence.running = true;
  const currentFile = current.current_file ? ` Current: ${current.current_file}.` : "";
  const message = `${current.completed || 0}/${current.total || 0} completed (${Number(current.percent || 0).toFixed(1)}%); ${current.succeeded || 0} succeeded, ${current.failed || 0} failed.${currentFile} Elapsed: ${Number(current.elapsed_seconds || 0).toFixed(0)} s.`;
  updateCurrentTaskProgress("Viability Detection Progress", message, Number(current.percent || 0), status.active_job_id, true);
  setAgentOutput(`[running] ${status.active_job_id}\n\n${message}`);
  setStopJobEnabled(true);
  if (!state.fluorescence.progressTimer) {
    state.fluorescence.progressTimer = setInterval(async () => {
      try {
        const refreshed = await fetchJson("/api/fluorescence/status");
        if (resumeFluorescenceProgress(refreshed)) return;
        state.fluorescence.running = false;
        state.fluorescence.progressTimer = null;
        updateCurrentTaskProgress("Viability Detection Progress", "No viability task is currently running.", null, "", false);
      } catch (_) {
        // Keep the last visible progress during a transient refresh failure.
      }
    }, 1000);
  }
  return true;
}

function updateCurrentTaskProgress(title, message, percent = null, jobId = "", active = true) {
  const heading = document.getElementById("current-task-title");
  const text = document.getElementById("current-task-progress");
  const bar = document.getElementById("current-task-progress-bar");
  const job = document.getElementById("current-task-job-id");
  if (heading) heading.textContent = title || "Current Task Progress";
  if (text) text.textContent = message || "No task running.";
  if (job) job.textContent = jobId || "";
  if (bar) {
    bar.hidden = !active;
    if (active && percent == null) bar.removeAttribute("value");
    if (active && percent != null) bar.value = Math.max(0, Math.min(100, Number(percent) || 0));
  }
}

function updateMorphologyProgressView(progress = {}, status = "pending") {
  updateCurrentTaskProgress("Morphology Analysis Progress", morphologyProgressText(progress, status), Number(progress.percent || 0), state.agent.morphologyJobId || "");
}

async function pollAgentJob(jobId) {
  if (!jobId) {
    return;
  }
  // A completed older task must never replace the progress or results of the
  // task the user selected most recently.
  if (state.agent.activeJobId !== jobId) return;
  try {
    const data = await fetchJson(`/api/agent/codex/job?id=${encodeURIComponent(jobId)}`);
    const job = data.job;
    const morphology = data.morphology_artifacts;
    const viability = data.viability_artifacts;
    const profiler = data.profiler_artifacts;
    const morphologyStatus = morphology?.status || "pending";
    const morphologyProgressData = morphology?.progress || {};
    const morphologyProgress = morphology?.total
      ? `\n\nMorphology ${morphology.managed_job_id || "job"}: ${morphologyProgressText(morphologyProgressData, morphologyStatus)}`
      : "";
    const profilerProgressData = profiler?.progress || {};
    const profilerProgress = profiler?.total
      ? `\n\n24D profiler ${profiler.job_id || "job"}: ${Number(profilerProgressData.completed || profiler.completed || 0)}/${Number(profilerProgressData.total || profiler.total || 0)} completed (${Number(profilerProgressData.percent || 0).toFixed(1)}%), status: ${profiler.status || "running"}.`
      : "";
    const body = (data.output_text || data.logs?.stderr_tail || data.logs?.stdout_tail || "Waiting for Codex output...") + morphologyProgress + profilerProgress;
    setAgentOutput(`[${job.status}] ${job.id}\n\n${body}`);
    const elapsed = state.agent.currentTaskStartedAt ? Math.max(0, (Date.now() - state.agent.currentTaskStartedAt) / 1000) : Number(job.elapsed_seconds || 0);
    const codexIsActive = ["queued", "running"].includes(job.status);
    const profilerCompleted = profiler?.status === "succeeded" && profiler?.rows?.length;
    const isResultSummaryJob = jobId === state.agent.viabilityAnalysisJobId || jobId === state.agent.morphologyAnalysisJobId || profilerCompleted;
    const progressTitle = isResultSummaryJob ? "AI Result Summary Progress" : `${pipelineScripts[state.agent.activeScript]?.title || "Analysis"} Progress`;
    const progressMessage = isResultSummaryJob
      ? `AI is generating the final table, interpretation, and requested chart. Status: ${job.status}. Elapsed: ${elapsed.toFixed(0)} s.`
      : `Agent task status: ${job.status}. Elapsed: ${elapsed.toFixed(0)} s.`;
    updateCurrentTaskProgress(progressTitle, progressMessage, null, job.id, codexIsActive);
    if (state.agent.activeScript === "organoid_analysis" || state.agent.activeScript === "fluorescence") {
      try {
        const viabilityStatus = await fetchJson("/api/fluorescence/status");
        const vp = viabilityStatus.progress;
        if (viabilityStatus.active_job_id && vp) {
          const currentFile = vp.current_file ? ` Current: ${vp.current_file}.` : "";
          updateCurrentTaskProgress("Viability Detection Progress", `${vp.completed || 0}/${vp.total || 0} completed (${Number(vp.percent || 0).toFixed(1)}%); ${vp.succeeded || 0} succeeded, ${vp.failed || 0} failed.${currentFile} Elapsed: ${Number(vp.elapsed_seconds || elapsed).toFixed(0)} s.`, Number(vp.percent || 0), viabilityStatus.active_job_id, true);
        }
      } catch (_) {
        // Keep the Agent's indeterminate progress visible until viability starts reporting.
      }
    }
    if (profiler?.job_id) {
      updateCurrentTaskProgress("24D Profiling Progress", `${Number(profilerProgressData.completed || profiler.completed || 0)}/${Number(profilerProgressData.total || profiler.total || 0)} completed (${Number(profilerProgressData.percent || 0).toFixed(1)}%). Status: ${profiler.status || "running"}.`, Number(profilerProgressData.percent || 0), profiler.job_id);
    }
    if (morphology?.managed_job_id) {
      state.agent.morphologyJobId = morphology.managed_job_id;
      state.agent.morphologyStatus = morphologyStatus;
      updateMorphologyProgressView(morphologyProgressData, morphologyStatus);
    }
    const codexTerminal = ["succeeded", "failed", "cancelled"].includes(job.status);
    const morphologyTerminal = ["succeeded", "failed", "cancelled"].includes(morphologyStatus);
    const waitingForMorphology = codexTerminal && morphology?.managed_job_id && !morphologyTerminal;
    if (waitingForMorphology) {
      setStopJobEnabled(true);
      setMorphologyRunning(true);
      loadAgentState();
      return;
    }
    if (codexTerminal) {
      clearInterval(state.agent.pollTimer);
      state.agent.pollTimer = null;
      setStopJobEnabled(false);
      setMorphologyRunning(false);
      if (!waitingForMorphology) updateCurrentTaskProgress(`${pipelineScripts[state.agent.activeScript]?.title || "Analysis"} Progress`, `Task ${job.status}.`, job.status === "succeeded" ? 100 : null, job.id, job.status === "succeeded");
      loadAgentState();
      if (morphology?.rows?.length) {
        renderMorphologyResults(morphology);
      }
      const isViabilityAnalysis = jobId === state.agent.viabilityAnalysisJobId
        || (job.tool === "response" && Boolean(viability?.samples?.length));
      if (viability?.samples?.length) {
        renderFluorescenceResults(viability);
        const viabilityReport = document.getElementById("fluorescence-report");
        if (viabilityReport && viability.report_url) {
          const response = await fetch(viability.report_url);
          viabilityReport.innerHTML = response.ok ? renderMarkdown(await response.text()) : "";
        }
        // A result-summary Agent job must not trigger an early second
        // navigation. The final navigation happens after its summary below.
        if (!isViabilityAnalysis) setActiveStage("results", "viability-results");
      }
      if (profiler?.rows?.length) {
        profiler.report_url = data.report_url || profiler.report_url || "";
        renderProfilerResults(profiler);
        if (profiler.report_url) {
          const reportResponse = await fetch(profiler.report_url);
          const report = document.getElementById("profiler-report");
          if (report) report.innerHTML = reportResponse.ok ? renderMarkdown(await reportResponse.text()) : "";
        }
        setActiveStage("results", "viability-results");
      }
      if (isViabilityAnalysis && data.output_text) {
        appendViabilityResultSummary(data.output_text);
        const viabilityRecord = state.results.records.find((record) => record.kind === "viability");
        if (viabilityRecord) {
          viabilityRecord.result_summary_output = data.output_text;
          viabilityRecord.result_summary_report_url = data.report_url || "";
          viabilityRecord.report_url = data.report_url || viabilityRecord.report_url || "";
        }
        setActiveStage("results", "viability-results");
        state.agent.viabilityAnalysisJobId = null;
      }
      if (isViabilityAnalysis && !state.results.recordedJobIds.has(job.id)) {
        appendAgentMessage("assistant", data.output_text || `Job ${job.status}: ${job.id}`);
        state.results.recordedJobIds.add(job.id);
        renderSessionResultsSummary();
        renderResultsHistory();
      }
      const isMorphologyAnalysis = jobId === state.agent.morphologyAnalysisJobId;
      if (isMorphologyAnalysis && data.output_text) {
        appendMorphologyResultSummary(data.output_text);
        const sourceId = state.agent.morphologyResultRecordId || (state.agent.morphologyJobId ? `morphology:${state.agent.morphologyJobId}` : "");
        const morphologyRecord = state.results.records.find((record) => record.id === sourceId)
          || state.results.records.find((record) => record.artifacts?.rows?.length);
        if (morphologyRecord) {
          morphologyRecord.result_summary_output = data.output_text;
          morphologyRecord.result_summary_report_url = data.report_url || "";
          morphologyRecord.report_url = data.report_url || morphologyRecord.report_url || "";
          morphologyRecord.status = "succeeded";
        }
        setActiveStage("results", "viability-results");
        state.agent.morphologyAnalysisJobId = null;
        renderSessionResultsSummary();
        renderResultsHistory();
      }
      // Result-summary Agent jobs belong to the already-created workflow run.
      // Keep their text/report on that run instead of displaying a second run.
      if (!state.results.recordedJobIds.has(job.id) && !isMorphologyAnalysis && !isViabilityAnalysis) {
        appendAgentMessage("assistant", data.output_text || `Job ${job.status}: ${job.id}`);
        state.results.recordedJobIds.add(job.id);
        const hasMorphology = Boolean(morphology?.managed_job_id || morphology?.rows?.length);
        const resultKind = profiler?.rows?.length ? "profiler" : (viability?.samples?.length && !hasMorphology ? "viability" : (hasMorphology ? "morphology" : "agent"));
        const recordId = `agent:${job.id}`;
        // An Agent morphology workflow has one user-facing run. The managed
        // morphology job is an execution child, so remove a child card that
        // may have been inserted while the parent was still running.
        if (hasMorphology && morphology.managed_job_id) {
          const childId = `morphology:${morphology.managed_job_id}`;
          state.results.records = state.results.records.filter((record) => record.id !== childId);
          state.agent.morphologyResultRecordId = recordId;
        }
        state.results.records.unshift({
          id: recordId,
          kind: resultKind,
          title: resultKind === "viability" ? `Viability · ${viability.job_id || job.id}` : (resultKind === "profiler" ? `24D Profile · ${profiler.job_id || job.id}` : (resultKind === "morphology" ? `Morphology Analysis · ${morphology.managed_job_id || job.id}` : `Agent · ${job.id}`)),
          status: morphology?.managed_job_id ? morphologyStatus : job.status,
          created_at: Date.now() / 1000,
          elapsed_seconds: job.elapsed_seconds,
          sample_count: viability?.samples?.length || 0,
          model: job.model,
          prompt_preview: job.prompt_preview || "",
          output_text: data.output_text || "",
          error: job.error,
          detail: job.detail,
          artifacts: morphology || {},
          payload: viability || null,
          profiler: profiler || null,
          report_url: data.report_url || "",
        });
        if (morphologyStatus === "succeeded" && morphology?.rows?.length) {
          setActiveStage("results", "viability-results");
        }
        renderSessionResultsSummary();
        renderResultsHistory();
      }
    }
  } catch (err) {
    setAgentOutput(`Poll failed: ${err.message}`);
  }
}

async function submitAgentJob(tool, prompt) {
  if (["zhengyu", "y27632", "sodium_alginate", "neo_cpsam"].includes(state.agent.activeScript)) {
    appendAgentMessage("user", prompt);
    appendAgentMessage("assistant", "Submitting the selected NEO images to the managed CellposeSAM workflow.");
    await runMorphologyDirect(prompt);
    return;
  }
  if (["organoid_analysis", "fluorescence"].includes(state.agent.activeScript)) {
    if (!state.agent.selectedFiles.size) {
      setAgentOutput("Select one or more images before starting viability detection.");
      updateCurrentTaskProgress("Viability Detection Progress", "Select one or more images before running this task.", null, "", false);
      return;
    }
    appendAgentMessage("user", prompt);
    appendAgentMessage("assistant", "Running the established YOLO + SAM + viability workflow on the selected images.");
    await runViabilityFromAgent(prompt);
    return;
  }
  if (state.agent.morphologyPollTimer) clearInterval(state.agent.morphologyPollTimer);
  state.agent.morphologyPollTimer = null;
  state.agent.morphologyJobId = null;
  state.agent.viabilityAnalysisJobId = null;
  showOnlyResultPanel();
  const pipelineText = document.getElementById("pipeline-editor")?.value || "";
  const sessionId = await ensureAgentSession();
  const profilerScale = document.getElementById("profiler-pixel-size")?.value.trim() || "";
  const routedPrompt = state.agent.activeScript === "organoid_profiler"
    ? `${prompt}\nProfiler pixel_size: ${profilerScale || "omit; use pixel units"}.`
    : prompt;
  const data = await postJson("/api/agent/chat", {
    session_id: sessionId,
    message: routedPrompt,
    tool,
    pipeline_text: pipelineText,
    allow_edits: tool === "assistant",
    workflow_id: state.agent.activeScript,
    selected_dataset: state.agent.selectedDataset,
    selected_files: Array.from(state.agent.selectedFiles),
  });
  state.agent.activeJobId = data.job.id;
  state.agent.currentTaskStartedAt = Date.now();
  setStopJobEnabled(true);
  appendAgentMessage("user", prompt);
  appendAgentMessage("assistant", `Started ${tool} job ${data.job.id}.`);
  setAgentOutput(`[queued] ${data.job.id}`);
  updateCurrentTaskProgress(`${pipelineScripts[state.agent.activeScript]?.title || "Analysis"} Progress`, "Task queued.", null, data.job.id);
  if (state.agent.pollTimer) {
    clearInterval(state.agent.pollTimer);
  }
  state.agent.pollTimer = setInterval(() => pollAgentJob(data.job.id), 2500);
  pollAgentJob(data.job.id);
}

async function startViabilityResultAnalysis(prompt, data) {
  const sessionId = await ensureAgentSession();
  const chartRequested = /图表|绘图|统计图|柱状图|折线图|趋势|chart|plot|graph|bar/i.test(prompt);
  const resultDirectory = `analysis-outputs/fluorescence_prediction/web/${data.job_id}`;
  let chartContext = "";
  if (chartRequested) {
    try {
      const chartData = await postJson("/api/charts/generate", {
        input: `${resultDirectory}/results.csv`,
        metrics: "viability",
      });
      chartContext = ` The workflow already generated the requested chart from the real CSV. Cite this web artifact exactly and do not call tools or generate another chart: ${JSON.stringify(chartData.charts || [])}.`;
    } catch (err) {
      chartContext = ` Chart generation failed before the summary: ${err.message}. State this exact failure and do not fabricate a chart.`;
    }
  }
  const measuredResults = JSON.stringify({
    summary: data.summary || {},
    samples: data.samples || [],
    failures: data.failures || [],
  });
  const followup = `The viability workflow has already completed successfully. This is the mandatory final AI result-summary step. Do not rerun segmentation or prediction. The exact measured workflow output is included below; use it as the authoritative source, together with ${resultDirectory}/results.json and ${resultDirectory}/results.csv when needed.\nMEASURED_WORKFLOW_OUTPUT=${measuredResults}\nFollow this original user instruction: ${prompt}\nThe workflow has already written the complete sorted results table above your section. Do not repeat or create a second table. Return only a concise interpretation section with: highest and lowest measured values, overall trend/classification counts, exact failures or manual-review items if present, and one chart link when a chart was requested. Use the supplied measured values; do not say that results are unavailable and do not add placeholder rows. Use web-relative URLs for evidence and chart links, never local filesystem paths. You may read the listed result JSON/CSV and evidence metadata, but do not edit repository files, run git, commit, or push.${chartRequested ? chartContext : " Do not generate a chart unless the user explicitly requested one."} Do not infer biological causality.`;
  const jobData = await postJson("/api/agent/chat", {
    session_id: sessionId,
    message: followup,
    tool: "response",
    job_purpose: "result_summary",
    reasoning: "medium",
    pipeline_text: pipelineScripts.organoid_analysis.text,
    allow_edits: false,
    workflow_id: "organoid_analysis",
    selected_dataset: state.agent.selectedDataset,
    selected_files: Array.from(state.agent.selectedFiles),
  });
  state.agent.activeJobId = jobData.job.id;
  state.agent.viabilityAnalysisJobId = jobData.job.id;
  state.agent.currentTaskStartedAt = Date.now();
  setStopJobEnabled(true);
  appendAgentMessage("assistant", `Viability prediction completed. Started result summarization ${jobData.job.id}.`);
  updateCurrentTaskProgress("Result Summary Progress", "Result summary queued.", null, jobData.job.id, true);
  if (state.agent.pollTimer) clearInterval(state.agent.pollTimer);
  state.agent.pollTimer = setInterval(() => pollAgentJob(jobData.job.id), 2500);
  pollAgentJob(jobData.job.id);
}

async function runViabilityFromAgent(instruction) {
  if (state.fluorescence.running) return;
  if (state.agent.pollTimer) clearInterval(state.agent.pollTimer);
  state.agent.pollTimer = null;
  if (state.agent.morphologyPollTimer) clearInterval(state.agent.morphologyPollTimer);
  state.agent.morphologyPollTimer = null;
  state.agent.activeJobId = null;
  state.agent.viabilityAnalysisJobId = null;
  state.agent.morphologyJobId = null;
  showOnlyResultPanel();
  state.fluorescence.running = true;
  state.agent.currentTaskStartedAt = Date.now();
  updateCurrentTaskProgress("Viability Detection Progress", `Submitting ${state.agent.selectedFiles.size} selected image(s)...`, 0, "", true);
  setAgentOutput("Submitting selected images to the established YOLO + SAM + viability workflow...");
  try {
    state.fluorescence.progressTimer = setInterval(async () => {
      try {
        const status = await fetchJson("/api/fluorescence/status");
        const current = status.progress;
        if (!current) return;
        const currentFile = current.current_file ? ` Current: ${current.current_file}.` : "";
        const message = `${current.completed || 0}/${current.total || 0} completed (${Number(current.percent || 0).toFixed(1)}%); ${current.succeeded || 0} succeeded, ${current.failed || 0} failed.${currentFile} Elapsed: ${Number(current.elapsed_seconds || 0).toFixed(0)} s.`;
        updateCurrentTaskProgress("Viability Detection Progress", message, Number(current.percent || 0), status.active_job_id || "", true);
        setAgentOutput(`[running] ${status.active_job_id || "viability"}\n\n${message}`);
      } catch (_) {
        // Preserve the last visible update during a transient status failure.
      }
    }, 1000);
    const data = await postJson("/api/fluorescence/run", {
      paths: Array.from(state.agent.selectedFiles),
      order: "desc",
      instruction,
      language: /[\u3400-\u9fff]/.test(instruction) ? "zh" : "auto",
    });
    renderFluorescenceResults(data);
    const reportResponse = await fetch(data.report_url);
    const report = document.getElementById("fluorescence-report");
    if (report) report.innerHTML = reportResponse.ok ? renderMarkdown(await reportResponse.text()) : "";
    updateCurrentTaskProgress("Viability Detection Progress", `Completed ${data.summary.succeeded} image(s); ${data.summary.failed} failed.`, 100, data.job_id, true);
    setAgentOutput(`[succeeded] ${data.job_id}\n\nCompleted ${data.summary.succeeded} image(s); ${data.summary.failed} failed.`);
    const recordId = `viability:${data.job_id}`;
    if (!state.results.records.some((item) => item.id === recordId)) {
      state.results.records.unshift({id: recordId, kind: "viability", title: `Viability · ${data.job_id}`, status: data.status || "succeeded", created_at: Date.now() / 1000, sample_count: data.samples?.length || 0, payload: data});
      state.agent.viabilityResultRecordId = recordId;
      state.results.selectedId = recordId;
      renderSessionResultsSummary();
      renderResultsHistory();
    }
    await startViabilityResultAnalysis(instruction, data);
  } catch (err) {
    updateCurrentTaskProgress("Viability Detection Progress", `Task failed: ${err.message}`, null, "", false);
    setAgentOutput(`Viability prediction failed: ${err.message}`);
    appendAgentMessage("assistant", `Viability prediction failed: ${err.message}`);
  } finally {
    if (state.fluorescence.progressTimer) clearInterval(state.fluorescence.progressTimer);
    state.fluorescence.progressTimer = null;
    state.fluorescence.running = false;
  }
}

function renderAgentFileSelection() {
  const container = document.getElementById("agent-file-selection");
  const count = document.getElementById("agent-selection-count");
  if (!container || !count) return;
  if (!state.agent.selectedDataset) {
    container.innerHTML = '<span class="muted">Select a dataset to analyze the whole dataset or choose specific images.</span>';
    count.textContent = "No dataset selected";
    return;
  }
  count.textContent = state.agent.selectedFiles.size
    ? `${state.agent.selectedFiles.size} image(s) selected`
    : "Whole dataset selected";
  if (!state.agent.availableFiles.length) {
    container.innerHTML = '<span class="muted">No supported images found. The dataset directory remains selected.</span>';
    return;
  }
  container.innerHTML = state.agent.availableFiles.map((file) => `
    <label class="agent-file-option">
      <input type="checkbox" value="${escapeHtml(file.path)}" ${state.agent.selectedFiles.has(file.path) ? "checked" : ""} />
      <span>${escapeHtml(file.path)}</span>
      <small>${escapeHtml(file.size_human || "")}</small>
    </label>
  `).join("");
  container.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.agent.selectedFiles.add(checkbox.value);
      else state.agent.selectedFiles.delete(checkbox.value);
      renderAgentFileSelection();
    });
  });
}

async function loadAgentDatasetFiles(dataset) {
  state.agent.selectedDataset = dataset;
  state.agent.selectedFiles.clear();
  state.agent.availableFiles = [];
  renderAgentFileSelection();
  if (!dataset) return;
  const workflowByDataset = {
    "01_Density_experiment_10x": "zhengyu",
    "02_Sodium_alginate_experiment_10x": "sodium_alginate",
    "03_Y-27632_experiment_10x": "y27632",
    "05_Fluorescence_demo": "organoid_analysis",
    "06_OrganoidProfiler_49": "organoid_profiler",
    "07_NEO": "neo_cpsam",
  };
  const datasetName = dataset.replace(/\\/g, "/").split("/").filter(Boolean).pop();
  const matchedWorkflow = workflowByDataset[datasetName];
  const scriptSelect = document.getElementById("script-select");
  if (matchedWorkflow && scriptSelect) {
    scriptSelect.value = matchedWorkflow;
    loadSelectedScript();
  }
  const data = await fetchJson(`/api/fluorescence/images?dataset=${encodeURIComponent(dataset)}`);
  state.agent.availableFiles = data.images || [];
  renderAgentFileSelection();
}

function initAgentDatasetSelector() {
  const select = document.getElementById("agent-dataset-select");
  if (!select) return;
  select.innerHTML = '<option value="">Select a dataset</option>' + state.datasets.map(
    (dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`
  ).join("");
  select.addEventListener("change", () => loadAgentDatasetFiles(select.value).catch((err) => {
    document.getElementById("agent-file-selection").textContent = err.message;
  }));
  document.getElementById("agent-select-all-files")?.addEventListener("click", () => {
    state.agent.selectedFiles = new Set(state.agent.availableFiles.map((file) => file.path));
    renderAgentFileSelection();
  });
  document.getElementById("agent-clear-files")?.addEventListener("click", () => {
    state.agent.selectedFiles.clear();
    renderAgentFileSelection();
  });
  renderAgentFileSelection();
}

function setMorphologyRunning(running) {
  const run = document.getElementById("morphology-run");
  const stop = document.getElementById("morphology-stop");
  if (run) run.disabled = running;
  if (stop) stop.disabled = !running;
}

function showOnlyResultPanel(panelId = "") {
  ["viability-results-card", "morphology-results-card", "profiler-results-card"].forEach((id) => {
    const panel = document.getElementById(id);
    if (panel) panel.hidden = id !== panelId;
  });
}

function appendViabilityResultSummary(markdown) {
  const report = document.getElementById("fluorescence-report");
  if (!report || !markdown) return;
  let section = report.querySelector("[data-viability-result-summary]");
  if (!section) {
    section = document.createElement("section");
    section.dataset.viabilityResultSummary = "true";
    section.className = "result-summary-analysis";
  report.insertBefore(section, report.firstChild);
  }
  section.innerHTML = `<hr><h3>Result Summary Table</h3>${renderMarkdown(markdown)}`;
}

function appendMorphologyResultSummary(markdown) {
  const report = document.getElementById("morphology-report");
  if (!report || !markdown) return;
  let section = report.querySelector("[data-morphology-result-summary]");
  if (!section) {
    section = document.createElement("section");
    section.dataset.morphologyResultSummary = "true";
    section.className = "result-summary-analysis";
  report.insertBefore(section, report.firstChild);
  }
  section.innerHTML = `<hr><h3>Result Summary</h3>${renderMarkdown(markdown)}`;
}

async function startMorphologyResultAnalysis(prompt, data) {
  if (!/(分析|解释|总结|汇总|表格|图表|绘图|趋势|analysis|analyze|explain|interpret|summary|table|chart|plot|graph)/i.test(prompt)) return;
  const sessionId = await ensureAgentSession();
  const chartRequested = /(图表|绘图|趋势|chart|plot|graph)/i.test(prompt);
  const resultDirectory = `analysis-outputs/morphology_jobs/${data.job_id}`;
  const workflowId = state.agent.activeScript;
  const followup = `The managed morphology workflow has completed successfully. Do not rerun segmentation. Analyze only the measured results in ${resultDirectory}/results.json (and results.csv when it exists). Follow this original user instruction: ${prompt}\nReturn the requested evidence-based result summary. When a table is requested, use a valid Markdown table and only measured fields.${chartRequested ? " Generate requested charts from the real CSV or JSON through http://127.0.0.1:8080/api/charts/generate and cite the artifact paths." : " Do not generate charts unless explicitly requested."} Preserve segmentation QC limitations and do not infer biological causality.`;
  const jobData = await postJson("/api/agent/chat", {
    session_id: sessionId,
    message: followup,
    tool: chartRequested ? "assistant" : "response",
    pipeline_text: pipelineScripts[workflowId]?.text || "",
    allow_edits: chartRequested,
    job_purpose: "result_summary",
    workflow_id: workflowId,
    selected_dataset: state.agent.selectedDataset,
    selected_files: Array.from(state.agent.selectedFiles),
  });
  state.agent.activeJobId = jobData.job.id;
  state.agent.morphologyAnalysisJobId = jobData.job.id;
  state.agent.currentTaskStartedAt = Date.now();
  setStopJobEnabled(true);
  updateCurrentTaskProgress("Result Summary Progress", "Result summary queued.", null, jobData.job.id, true);
  if (state.agent.pollTimer) clearInterval(state.agent.pollTimer);
  state.agent.pollTimer = setInterval(() => pollAgentJob(jobData.job.id), 2500);
  pollAgentJob(jobData.job.id);
}

function renderMorphologyResults(data) {
  if (!data) return;
  showOnlyResultPanel("morphology-results-card");
  const card = document.getElementById("morphology-results-card");
  const summary = document.getElementById("morphology-summary");
  const results = document.getElementById("morphology-results");
  const report = document.getElementById("morphology-report");
  card.hidden = false;
  document.getElementById("morphology-result-job-id").textContent = data.job_id || "";
  const downloads = data.download_urls || {};
  summary.innerHTML = `
    <p class="hint">Run-level review is shown above. Scroll through the image cards below and use <strong>Review this image</strong> for a single image.</p>
    <div class="status-grid">
      <div class="status-row"><span>Images</span><strong>${(data.rows || []).length}</strong></div>
      <div class="status-row"><span>GPU</span><strong>${data.used_gpu ? "CUDA" : "CPU"}</strong></div>
      <div class="status-row"><span>Workflow</span><strong>${escapeHtml(data.workflow || "")}</strong></div>
      <div class="status-row"><span>Downloads</span><strong>${downloads.csv ? `<a href="${escapeHtml(downloads.csv)}" target="_blank">CSV</a>` : ""} ${downloads.json ? `· <a href="${escapeHtml(downloads.json)}" target="_blank">JSON</a>` : ""} ${downloads.report ? `· <a href="${escapeHtml(downloads.report)}" target="_blank">Report</a>` : ""}</strong></div>
    </div>
    ${(downloads.gallery || downloads.summary_figure) ? `<div class="fluorescence-images">${downloads.gallery ? `<figure><img src="${escapeHtml(downloads.gallery)}" alt="Morphology comparison" /><figcaption>Comparison gallery</figcaption></figure>` : ""}${downloads.summary_figure ? `<figure><img src="${escapeHtml(downloads.summary_figure)}" alt="Morphology metrics" /><figcaption>Summary metrics</figcaption></figure>` : ""}</div>` : ""}`;
  results.innerHTML = (data.rows || []).map((row) => {
    const assets = row.asset_urls || {};
    const targetKey = row.source_tif || row.image_name || "";
    const targetLabel = row.image_name || row.source_tif || "image";
    return `<article class="fluorescence-result">
      <div class="fluorescence-result-head"><div><strong>${escapeHtml(row.condition)} · D${String(row.day).padStart(2, "0")}</strong><div class="muted">${escapeHtml(row.image_name || "")}</div></div><div><div class="viability-score">${Number(row.count || 0)} objects</div><button class="soft-btn image-review-btn" data-review-job="${escapeHtml(data.job_id || "")}" data-review-key="${escapeHtml(targetKey)}" data-review-label="${escapeHtml(targetLabel)}" data-review-path="${escapeHtml(row.source_tif || "")}">Review this image</button></div></div>
      <div class="fluorescence-images">
        ${assets.overlay ? `<figure><img src="${escapeHtml(assets.overlay)}" alt="Selected best overlay" /><figcaption>Selected best segmentation</figcaption></figure>` : ""}
        ${assets.mask ? `<figure><img src="${escapeHtml(assets.mask)}" alt="Selected best mask" /><figcaption>Selected best instance mask</figcaption></figure>` : ""}
      </div>
      <div class="fluorescence-metrics">
        <div class="fluorescence-metric"><span>Total area</span><strong>${Number(row.total_area_px || 0).toLocaleString()}</strong></div>
        <div class="fluorescence-metric"><span>Mean area</span><strong>${Number(row.mean_area_px || 0).toLocaleString(undefined, {maximumFractionDigits: 1})} px²</strong></div>
        <div class="fluorescence-metric"><span>Roundness</span><strong>${Number(row.roundness || 0).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Largest mass fraction</span><strong>${Number(row.fused_mass_fraction || 0).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Mean diameter</span><strong>${Number(row.mean_equivalent_diameter_px || 0).toFixed(1)} px</strong></div>
        <div class="fluorescence-metric"><span>Stage</span><strong>${escapeHtml(row.stage || "")}</strong></div>
        <div class="fluorescence-metric"><span>Selection</span><strong>${escapeHtml(row.segmentation_method || "legacy")}</strong></div>
        <div class="fluorescence-metric"><span>Quality score</span><strong>${row.segmentation_quality_score == null ? "NA (original pipeline)" : Number(row.segmentation_quality_score).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Decision</span><strong>${escapeHtml(row.segmentation_selection_reason || "not recorded")}</strong></div>
      </div>
    </article>`;
  }).join("");
  results.querySelectorAll("[data-review-job]").forEach((button) => {
    button.addEventListener("click", () => mountReviewRecord(button.dataset.reviewJob, button.dataset.reviewLabel, button.dataset.reviewKey, button.dataset.reviewPath));
  });
  report.innerHTML = renderMarkdown(data.report_text || "");
}

function renderProfilerResults(data) {
  showOnlyResultPanel("profiler-results-card");
  const card = document.getElementById("profiler-results-card");
  const summary = document.getElementById("profiler-summary");
  const results = document.getElementById("profiler-results");
  if (!card || !summary || !results) return;
  card.hidden = false;
  document.getElementById("profiler-result-job-id").textContent = data.job_id || "";
  const chartLinks = (data.chart_urls || []).map((chart) => `<a href="${escapeHtml(chart.url)}" target="_blank">${escapeHtml(chart.metric)} line chart</a>`).join(" · ");
  summary.innerHTML = `<div class="status-grid"><div class="status-row"><span>Images</span><strong>${(data.rows || []).length}</strong></div><div class="status-row"><span>Units</span><strong>${escapeHtml(data.units || "pixel")}</strong></div><div class="status-row"><span>Pixel size</span><strong>${data.pixel_size == null ? "not supplied" : escapeHtml(String(data.pixel_size))}</strong></div><div class="status-row"><span>Downloads</span><strong><a href="${escapeHtml(data.csv_url || "#")}" target="_blank">CSV</a> · <a href="${escapeHtml(data.results_url || "#")}" target="_blank">JSON</a>${chartLinks ? ` · ${chartLinks}` : ""}</strong></div></div>`;
  const chartImages = (data.chart_urls || []).map((chart) => `<figure><img src="${escapeHtml(chart.url)}" alt="${escapeHtml(chart.metric)} line chart" /><figcaption>${escapeHtml(chart.metric)} longitudinal line chart</figcaption></figure>`).join("");
  if (chartImages) summary.innerHTML += `<div class="fluorescence-images">${chartImages}</div>`;
  const shown = ["growth_rate", "area", "equivalent_circle_diameter", "circularity", "roundness", "aspect_ratio", "solidity", "eccentricity", "perimeter", "ellipse_major_axis", "ellipse_minor_axis", "feret_maximum", "feret_minimum", "mean_intensity", "median_intensity", "mode_intensity", "intensity_std", "corrected_total_intensity", "corrected_mean_intensity", "integrated_intensity", "skewness", "kurtosis", "centroid_to_center_of_mass"];
  results.innerHTML = (data.rows || []).map((row, index) => {
    const metrics = shown.map((key) => `<div class="fluorescence-metric"><span>${escapeHtml(key)}</span><strong>${row[key] == null ? "NA" : Number(row[key]).toFixed(4)}</strong></div>`).join("");
    const label = `${row.image_name || "image"} · D${row.day || "NA"} · group ${row.group || "NA"}`;
    return `<article class="fluorescence-result"><div class="fluorescence-result-head"><div><strong>#${index + 1} ${escapeHtml(label)}</strong><div class="muted">YOLO ${Number(row.yolo_confidence || 0).toFixed(4)} · SAM ${Number(row.sam_score || 0).toFixed(4)}</div></div><button class="soft-btn image-review-btn" data-review-job="${escapeHtml(data.job_id || "")}" data-review-key="${escapeHtml(row.source_path || row.image_name || "")}" data-review-label="${escapeHtml(row.image_name || "image")}" data-review-path="${escapeHtml(row.source_path || "")}">Review this image</button></div><div class="fluorescence-images">${row.asset_urls?.overlay ? `<figure><img src="${escapeHtml(row.asset_urls.overlay)}" alt="Profiler overlay" /><figcaption>YOLO + SAM overlay</figcaption></figure>` : ""}${row.asset_urls?.mask ? `<figure><img src="${escapeHtml(row.asset_urls.mask)}" alt="Profiler mask" /><figcaption>Mask</figcaption></figure>` : ""}</div><div class="fluorescence-metrics">${metrics}</div></article>`;
  }).join("");
  results.querySelectorAll("[data-review-job]").forEach((button) => button.addEventListener("click", () => mountReviewRecord(button.dataset.reviewJob, button.dataset.reviewLabel, button.dataset.reviewKey, button.dataset.reviewPath)));
}

async function pollMorphologyJob() {
  const jobId = state.agent.morphologyJobId;
  if (!jobId) return;
  const text = document.getElementById("current-task-progress");
  try {
    const data = await fetchJson(`/api/morphology/status?id=${encodeURIComponent(jobId)}`);
    if (state.agent.morphologyJobId !== jobId) return;
    const job = data.job || {};
    const progress = data.progress || {};
    if (job.started_at) {
      progress.elapsed_seconds = Math.max(Number(progress.elapsed_seconds || 0), Date.now() / 1000 - Number(job.started_at));
    }
    state.agent.morphologyStatus = job.status || null;
    if (!progress.total && job.inputs?.length) progress.total = job.inputs.length;
    updateMorphologyProgressView(progress, job.status);
    setAgentOutput(`[${job.status}] Managed morphology job ${jobId}\n\n${morphologyProgressText(progress, job.status)}${data.stderr_tail ? `\n\n${data.stderr_tail}` : ""}`);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      clearInterval(state.agent.morphologyPollTimer);
      state.agent.morphologyPollTimer = null;
      setMorphologyRunning(false);
      setStopJobEnabled(false);
      loadAgentState();
      if (text && job.status !== "succeeded") text.textContent += ` ${job.error || data.stderr_tail || ""}`;
      if (job.status === "succeeded" && data.result) {
        renderMorphologyResults(data.result);
      }
      // When morphology was submitted by an Agent workflow, this is an
      // internal child job. Its artifacts are picked up by pollAgentJob and
      // displayed inside the single parent morphology-analysis record.
      if (job.parent_job_id) {
        if (job.status === "succeeded" && data.result) {
          state.agent.morphologyStatus = job.status;
        }
        return;
      }
      const recordId = `morphology:${jobId}`;
      if (!state.results.records.some((item) => item.id === recordId)) {
        state.results.records.unshift({id: recordId, kind: "morphology", title: `Morphology · ${jobId}`, status: job.status, created_at: job.created_at, elapsed_seconds: progress.elapsed_seconds, output_text: data.stdout_tail || data.stderr_tail || "", artifacts: data.result || {output_dir: job.output_dir}});
        state.agent.morphologyResultRecordId = recordId;
        renderSessionResultsSummary();
        renderResultsHistory();
      }
      if (job.status === "succeeded" && data.result) {
        const requestedSummary = /(分析|解释|总结|汇总|表格|图表|绘图|趋势|analysis|analyze|explain|interpret|summary|table|chart|plot|graph)/i.test(state.agent.morphologyInstruction || "")
          || state.agent.activeScript === "neo_cpsam";
        if (requestedSummary) {
          await startMorphologyResultAnalysis(state.agent.morphologyInstruction || "", data.result);
        } else {
          setActiveStage("results", "viability-results");
        }
      }
    }
  } catch (err) {
    if (text) text.textContent = `Morphology status failed: ${err.message}`;
    setAgentOutput(`Morphology status failed: ${err.message}`);
  }
}

async function runMorphologyDirect(instructionOverride = "") {
  const text = document.getElementById("current-task-progress");
  const instruction = instructionOverride.trim() || "Analyze the selected images.";
  if (!["zhengyu", "y27632", "sodium_alginate", "neo_cpsam"].includes(state.agent.activeScript)) {
    if (text) text.textContent = `Delegated ${pipelineScripts[state.agent.activeScript]?.title || "selected workflow"} to the Agent with the selected inputs and instruction.`;
    submitAgentJob("assistant", instruction);
    return;
  }
  if (!state.agent.selectedFiles.size) {
    if (text) text.textContent = "Select one or more images. Whole-dataset execution is disabled for this long-running workflow.";
    return;
  }
  if (state.agent.pollTimer) clearInterval(state.agent.pollTimer);
  state.agent.pollTimer = null;
  if (state.agent.morphologyPollTimer) clearInterval(state.agent.morphologyPollTimer);
  state.agent.morphologyPollTimer = null;
  state.agent.activeJobId = null;
  state.agent.viabilityAnalysisJobId = null;
  state.agent.morphologyAnalysisJobId = null;
  state.agent.morphologyInstruction = instruction;
  showOnlyResultPanel();
  setMorphologyRunning(true);
  updateCurrentTaskProgress("Morphology Analysis Progress", "Submitting morphology job...", 0);
  setAgentOutput("Submitting selected images to the managed morphology workflow...");
  try {
    const data = await postJson("/api/morphology/run", {paths: Array.from(state.agent.selectedFiles), overwrite: false, workflow_id: state.agent.activeScript, instruction});
    state.agent.morphologyJobId = data.job_id;
    state.agent.morphologyStatus = data.status || "queued";
    setStopJobEnabled(true);
    setAgentOutput(`[${state.agent.morphologyStatus}] Managed morphology job ${data.job_id}\n\nWaiting for the first progress update...`);
    updateCurrentTaskProgress("Morphology Analysis Progress", "Waiting for the first progress update...", 0, data.job_id);
    state.agent.morphologyPollTimer = setInterval(pollMorphologyJob, 1500);
    pollMorphologyJob();
  } catch (err) {
    setMorphologyRunning(false);
    if (text) text.textContent = `Morphology submission failed: ${err.message}`;
    setAgentOutput(`Morphology submission failed: ${err.message}`);
  }
}

async function stopMorphologyDirect() {
  if (!state.agent.morphologyJobId) return;
  const text = document.getElementById("current-task-progress");
  try {
    await postJson("/api/morphology/cancel", {job_id: state.agent.morphologyJobId});
    if (text) text.textContent = "Stopping morphology job...";
    pollMorphologyJob();
  } catch (err) {
    if (text) text.textContent = `Stop failed: ${err.message}`;
  }
}

function insertPipelineTemplate(kind) {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const snippets = {
    inspect: 'STEP {"id":"inspect_next","block":"plan","title":"Inspect inputs","instruction":"Identify source folders, channels, time/depth design, references, scripts, and output requirements."}\nACTION {"type":"read","target":"references"}',
    segmentation: 'STEP {"id":"segment_next","block":"work","title":"Segment brightfield","instruction":"Run the configured Cellpose-first morphology segmentation and save selected masks, overlays, and quality evidence."}\nACTION {"type":"api","target":"/api/morphology/run"}',
    tracking: 'STEP {"id":"tracking_next","block":"work","title":"Track positions over time","instruction":"Link objects across day/time/position folders and record monitoring metadata."}\nACTION {"type":"database","target":"analysis-outputs"}',
    pairing: 'STEP {"id":"pair_next","block":"work","title":"Create paired crops","instruction":"Match selected image channels only when explicitly requested and write paired records under analysis-outputs."}\nACTION {"type":"dataset","target":"analysis-outputs"}',
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
  document.getElementById("stop-agent-job")?.addEventListener("click", async () => {
    const morphologyIsActive = state.agent.morphologyJobId && ["queued", "running", "pending"].includes(state.agent.morphologyStatus);
    const fluorescenceIsActive = state.fluorescence.running;
    if (!morphologyIsActive && !state.agent.activeJobId && !fluorescenceIsActive) return;
    const button = document.getElementById("stop-agent-job");
    button.disabled = true;
    button.textContent = "Stopping...";
    try {
      if (morphologyIsActive) {
        const data = await postJson("/api/morphology/cancel", { job_id: state.agent.morphologyJobId });
        state.agent.morphologyStatus = data.job.status;
        setAgentOutput(`[${data.job.status}] ${data.job.id || state.agent.morphologyJobId}\n\nMorphology job stopped by user.`);
        pollAgentJob(state.agent.activeJobId);
        loadAgentState();
        return;
      }
      if (fluorescenceIsActive && !state.agent.activeJobId) {
        const status = await fetchJson("/api/fluorescence/status");
        const data = await postJson("/api/fluorescence/cancel", { job_id: status.active_job_id || "" });
        state.fluorescence.running = false;
        if (state.fluorescence.progressTimer) clearInterval(state.fluorescence.progressTimer);
        state.fluorescence.progressTimer = null;
        setAgentOutput(`[${data.status}] ${data.job_id}\n\nViability detection stopped by user.`);
        updateCurrentTaskProgress("Viability Detection Progress", "Viability detection stopped by user.", null, data.job_id, false);
        loadAgentState();
        return;
      }
      const data = await postJson("/api/agent/codex/cancel", { job_id: state.agent.activeJobId });
      setAgentOutput(`[${data.job.status}] ${data.job.id}\n\nJob stopped by user.`);
      pollAgentJob(state.agent.activeJobId);
    } catch (err) {
      setAgentOutput(`Stop failed: ${err.message}`);
      setStopJobEnabled(true);
    } finally {
      button.textContent = "Stop Job";
    }
  });
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
  }, data.empty_message || "No items found.");
}

async function loadSegmentationCategory() {
  const data = await fetchJson("/api/category/segmentation");
  const files = data.files || [];
  const groups = new Map();
  files.forEach((file) => {
    const key = `${file.dataset || "Generated output"}::${file.group || "Unassigned run"}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        name: file.group || "Unassigned run",
        dataset: file.dataset || "Generated output",
        kind: "analysis run",
        files: [],
      });
    }
    groups.get(key).files.push(file);
  });
  const entries = Array.from(groups.values());
  const renderAssets = (group) => {
    renderList("segmentation-list", group.files, async (file) => {
      const preview = await fetchJson(`/api/preview?path=${encodeURIComponent(file.path)}`);
      renderPreview("segmentation-preview", preview);
    }, "No masks or overlays were found in this run.");
    const preview = document.getElementById("segmentation-preview");
    if (preview) preview.textContent = `Select an asset from ${group.dataset} · ${group.name}.`;
  };
  renderList("segmentation-group-list", entries, renderAssets, data.empty_message || "No segmentation runs found.");
  if (entries.length) renderAssets(entries[0]);
}

function consolidateMorphologyRecords() {
  // Repair sessions created before the parent/child aggregation fix loaded.
  const managedParents = state.results.records.filter((record) => record.kind === "agent" && record.artifacts?.managed_job_id);
  managedParents.forEach((parent) => {
    const managedId = parent.artifacts.managed_job_id;
    parent.kind = "morphology";
    parent.title = `Morphology Analysis · ${managedId}`;
    state.results.records = state.results.records.filter((record) => record.id !== `morphology:${managedId}` || record.id === parent.id);
    state.agent.morphologyResultRecordId = parent.id;
  });
}

function renderResultsHistory() {
  const list = document.getElementById("results-history-list");
  if (!list) return;
  consolidateMorphologyRecords();
  if (!state.results.records.length) {
    list.innerHTML = '<div class="muted">No tasks have been run on this page yet.</div>';
    return;
  }
  list.innerHTML = state.results.records.map((record) => {
    const active = record.id === state.results.selectedId ? " active" : "";
    const date = record.created_at ? new Date(record.created_at * 1000).toLocaleString() : "Unknown time";
    const detail = record.kind === "viability"
      ? `${record.sample_count} sample(s)`
      : `${escapeHtml(record.model || "Codex")} · ${Number(record.elapsed_seconds || 0).toFixed(1)} s`;
    return `
      <button class="result-history-item${active}" data-result-id="${escapeHtml(record.id)}">
        <span><strong>${escapeHtml(record.title)}</strong><small>${escapeHtml(date)}</small></span>
        <span><span class="result-status ${escapeHtml(record.status)}">${escapeHtml(record.status)}</span><small>${detail}</small></span>
      </button>`;
  }).join("");
  list.querySelectorAll("[data-result-id]").forEach((button) => {
    button.addEventListener("click", () => openResultRecord(button.dataset.resultId));
  });
}

async function openResultRecord(recordId) {
  const record = state.results.records.find((item) => item.id === recordId);
  if (!record) return;
  state.results.selectedId = recordId;
  renderResultsHistory();
  showOnlyResultPanel();
  const detail = document.getElementById("results-history-detail");
  if (record.kind === "profiler") {
    const payload = record.profiler;
    detail.innerHTML = `<h3>${escapeHtml(record.title)}</h3><p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Images:</strong> ${(payload?.rows || []).length}</p><p>${payload?.results_url ? `<a href="${escapeHtml(payload.results_url)}" target="_blank">JSON</a>` : ""}${payload?.csv_url ? ` · <a href="${escapeHtml(payload.csv_url)}" target="_blank">CSV</a>` : ""}${record.report_url ? ` · <a href="${escapeHtml(record.report_url)}" target="_blank">Result report</a>` : ""}</p>`;
    mountReviewRecord(payload?.job_id || record.id, record.title);
    renderProfilerResults(payload);
    if (record.report_url) {
      const reportResponse = await fetch(record.report_url);
      const report = document.getElementById("profiler-report");
      if (report) report.innerHTML = reportResponse.ok ? renderMarkdown(await reportResponse.text()) : "";
    }
    return;
  }
  if (record.kind === "viability") {
    const payload = record.payload;
    const reviewJobId = payload.job_id || record.id;
    detail.innerHTML = `
      <h3>${escapeHtml(record.title)}</h3>
      <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Samples:</strong> ${record.sample_count}</p>
      <p><strong>Run:</strong> ${escapeHtml(payload.job_id || record.id)}</p>
      <p><a href="${escapeHtml(payload.results_url)}" target="_blank">JSON</a>${payload.csv_url ? ` · <a href="${escapeHtml(payload.csv_url)}" target="_blank">CSV</a>` : ""}${payload.report_url ? ` · <a href="${escapeHtml(payload.report_url)}" target="_blank">Report</a>` : ""}</p>`;
    mountReviewRecord(reviewJobId, record.title);
    renderFluorescenceResults(payload);
    document.getElementById("fluorescence-progress").textContent = `Loaded saved run ${record.title}.`;
    if (payload.report_url) {
      const response = await fetch(payload.report_url);
      document.getElementById("fluorescence-report").innerHTML = response.ok ? renderMarkdown(await response.text()) : "";
    }
    if (record.result_summary_output) appendViabilityResultSummary(record.result_summary_output);
    return;
  }
  if (record.artifacts?.rows?.length) {
    const morphology = record.artifacts;
    const reviewJobId = morphology.job_id || record.id;
      const downloads = morphology.download_urls || {};
      detail.innerHTML = `
      <h3>${escapeHtml(record.title)}</h3>
      <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Morphology outputs:</strong> ${morphology.completed || morphology.rows.length}/${morphology.total || morphology.rows.length}</p>
      <p>${downloads.json ? `<a href="${escapeHtml(downloads.json)}" target="_blank">JSON</a>` : ""}${downloads.csv ? ` · <a href="${escapeHtml(downloads.csv)}" target="_blank">CSV</a>` : ""}${downloads.report ? ` · <a href="${escapeHtml(downloads.report)}" target="_blank">Segmentation report</a>` : ""}${record.result_summary_report_url || record.report_url ? ` · <a href="${escapeHtml(record.result_summary_report_url || record.report_url)}" target="_blank">Agent report</a>` : ""}</p>
      <pre class="result-output-text">${escapeHtml(record.output_text || "")}</pre>`;
    mountReviewRecord(reviewJobId, record.title);
    renderMorphologyResults(morphology);
    if (record.result_summary_output) appendMorphologyResultSummary(record.result_summary_output);
    return;
  }
  const links = Object.entries(record.artifacts || {}).filter(([, url]) => url).map(
    ([name, url]) => `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(name)}</a>`
  ).join(" · ");
  detail.innerHTML = `
    <h3>${escapeHtml(record.title)}</h3>
    <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Model:</strong> ${escapeHtml(record.model || "")}</p>
    <p><strong>Prompt:</strong> ${escapeHtml(record.prompt_preview || "")}</p>
    ${record.error ? `<p class="error"><strong>Error:</strong> ${escapeHtml(record.error)} ${escapeHtml(record.detail || "")}</p>` : ""}
    <p>${links}</p>
    <pre class="result-output-text">${escapeHtml(record.output_text || "No output text was saved.")}</pre>`;
  mountReviewRecord(record.id, record.title);
}

const reviewFlagOptions = [
  ["debris", "Debris"], ["edge_clipping", "Edge clipping"], ["bad_focus", "Bad focus"],
  ["wrong_channel", "Wrong channel"], ["bad_pairing", "Bad pairing"],
  ["low_confidence", "Low confidence"], ["mask_error", "Mask error"], ["other", "Other"],
];

async function mountReviewRecord(jobId, title, targetKey = "", targetPath = "") {
  const panel = document.getElementById("review-record-panel");
  const jobLabel = document.getElementById("review-record-job");
  if (!panel || !jobId) return;
  jobLabel.textContent = targetKey ? `${jobId} · ${title}` : jobId;
  panel.textContent = "Loading review records…";
  const targetQuery = targetKey ? `&target_key=${encodeURIComponent(targetKey)}` : "";
  const data = await fetchJson(`/api/reviews?job_id=${encodeURIComponent(jobId)}${targetQuery}`);
  const latest = data.reviews?.[0] || {};
  panel.innerHTML = `
    <div class="status-grid"><div><span>Target</span><strong>${escapeHtml(title || jobId)}</strong></div>
    <div><span>Existing records</span><strong>${data.reviews?.length || 0}</strong></div></div>
    <div class="agent-data-controls">
      <label>Status <select id="review-status">${["pending", "approved", "rejected", "needs_correction"].map((s) => `<option value="${s}" ${latest.status === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
      <label>Reviewer <input id="reviewer-name" value="${escapeHtml(latest.reviewer || "")}" maxlength="160" /></label>
    </div>
    <div class="agent-file-selection">${reviewFlagOptions.map(([value, label]) => `<label class="agent-file-option"><input type="checkbox" value="${value}" class="review-flag" ${(latest.flags || []).includes(value) ? "checked" : ""} /> ${label}</label>`).join("")}</div>
    <textarea id="review-notes" rows="4" maxlength="4000" placeholder="Review notes">${escapeHtml(latest.notes || "")}</textarea>
    <button class="soft-btn" id="save-review-record">Save review record</button>
    <div class="muted" id="review-save-status"></div>`;
  document.getElementById("save-review-record")?.addEventListener("click", async () => {
    const status = document.getElementById("review-save-status");
    status.textContent = "Saving…";
    try {
      const saved = await fetchJson("/api/reviews", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
        review_id: latest.review_id || "", job_id: jobId, target_label: title || jobId,
        target_key: targetKey, target_path: targetPath, status: document.getElementById("review-status").value,
        reviewer: document.getElementById("reviewer-name").value,
        notes: document.getElementById("review-notes").value,
        flags: Array.from(document.querySelectorAll(".review-flag:checked")).map((el) => el.value),
      })});
      status.textContent = `Saved ${saved.status} at ${new Date(saved.updated_at * 1000).toLocaleString()}.`;
    } catch (err) { status.textContent = `Save failed: ${err.message}`; }
  });
}

function renderSessionResultsSummary() {
  consolidateMorphologyRecords();
  const summary = {
    total: state.results.records.length,
    viability: state.results.records.filter((item) => item.kind === "viability").length,
    morphology: state.results.records.filter((item) => item.kind !== "viability").length,
    succeeded: state.results.records.filter((item) => item.status === "succeeded").length,
    failed: state.results.records.filter((item) => item.status === "failed").length,
  };
  document.getElementById("results-history-summary").innerHTML = `
    <div><span>All runs</span><strong>${summary.total || 0}</strong></div>
    <div><span>Viability</span><strong>${summary.viability || 0}</strong></div>
    <div><span>Analysis jobs</span><strong>${summary.morphology || 0}</strong></div>
    <div><span>Succeeded / Failed</span><strong>${summary.succeeded || 0} / ${summary.failed || 0}</strong></div>`;
}

function updateFluorescenceSelectionCount() {
  const count = state.fluorescence.selected.size;
  const target = document.getElementById("fluorescence-selection-count");
  if (target) {
    target.textContent = `${count} selected`;
  }
}

function renderFluorescenceImages() {
  const container = document.getElementById("fluorescence-image-list");
  if (!container) {
    return;
  }
  if (!state.fluorescence.images.length) {
    container.innerHTML = "<div class='muted' style='padding:12px'>No supported images found.</div>";
    updateFluorescenceSelectionCount();
    return;
  }
  container.innerHTML = state.fluorescence.images
    .map((item) => `
      <label class="fluorescence-image-option">
        <input type="checkbox" data-fluorescence-path="${escapeHtml(item.path)}" ${state.fluorescence.selected.has(item.path) ? "checked" : ""} />
        <span>${escapeHtml(item.path)}</span>
        <small class="muted">${escapeHtml(item.size_human)}</small>
      </label>
    `)
    .join("");
  container.querySelectorAll("[data-fluorescence-path]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        if (state.fluorescence.selected.size >= 30) {
          checkbox.checked = false;
          return;
        }
        state.fluorescence.selected.add(checkbox.dataset.fluorescencePath);
      } else {
        state.fluorescence.selected.delete(checkbox.dataset.fluorescencePath);
      }
      updateFluorescenceSelectionCount();
    });
  });
  updateFluorescenceSelectionCount();
}

async function loadFluorescenceImages(dataset) {
  const container = document.getElementById("fluorescence-image-list");
  container.textContent = "Loading images...";
  state.fluorescence.selected.clear();
  const data = await fetchJson(`/api/fluorescence/images?dataset=${encodeURIComponent(dataset)}`);
  state.fluorescence.images = data.images || [];
  renderFluorescenceImages();
}

function renderFluorescenceResults(data) {
  showOnlyResultPanel("viability-results-card");
  const summary = document.getElementById("fluorescence-summary");
  const results = document.getElementById("fluorescence-results");
  const job = document.getElementById("fluorescence-job-id");
  job.textContent = data.job_id || "";
  summary.innerHTML = `
    <div class="status-grid">
      <div class="status-row"><span>Successful</span><strong>${data.summary.succeeded}/${data.summary.total}</strong></div>
      <div class="status-row"><span>High / Medium / Low</span><strong>${data.summary.high} / ${data.summary.medium} / ${data.summary.low}</strong></div>
      <div class="status-row"><span>Elapsed</span><strong>${Number(data.summary.elapsed_seconds).toFixed(1)} s</strong></div>
      <div class="status-row"><span>Downloads</span><strong><a href="${escapeHtml(data.csv_url)}" target="_blank">CSV</a> · <a href="${escapeHtml(data.results_url)}" target="_blank">JSON</a></strong></div>
    </div>
  `;
  results.innerHTML = (data.samples || []).map((sample, index) => {
    const e = sample.evidence;
    return `
      <article class="fluorescence-result">
        <div class="fluorescence-result-head">
          <div><strong>#${index + 1} ${escapeHtml(sample.filename)}</strong><div class="muted">${escapeHtml(sample.level)} · YOLO ${sample.segmentation.yolo_confidence} · SAM ${sample.segmentation.sam_score}</div></div>
          <div><div class="viability-score">${Number(sample.viability_percent).toFixed(1)}%</div><button class="soft-btn image-review-btn" data-review-job="${escapeHtml(data.job_id || "")}" data-review-key="${escapeHtml(sample.source_path || sample.filename || "")}" data-review-label="${escapeHtml(sample.filename || "image")}" data-review-path="${escapeHtml(sample.source_path || "")}">Review this image</button></div>
        </div>
        <div class="fluorescence-images">
          <figure><img src="${escapeHtml(sample.asset_urls.overlay)}" alt="Segmentation overlay" /><figcaption>YOLO + SAM overlay</figcaption></figure>
          <figure><img src="${escapeHtml(sample.asset_urls.crop)}" alt="Prediction crop" /><figcaption>Model input crop</figcaption></figure>
        </div>
        <div class="fluorescence-metrics">
          <div class="fluorescence-metric"><span>Circularity</span><strong>${Number(e.circularity).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Area fraction</span><strong>${Number(e.area_fraction).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Bounding-box fill</span><strong>${Number(e.bbox_fill_ratio).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Internal contrast</span><strong>${Number(e.contrast).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Edge density</span><strong>${Number(e.edge_density).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Mean brightness</span><strong>${Number(e.mean_brightness).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Aspect ratio</span><strong>${Number(e.aspect_ratio).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Processing time</span><strong>${Number(sample.elapsed_seconds).toFixed(1)} s</strong></div>
        </div>
        <p>${escapeHtml(sample.explanation)}</p>
        <a href="${escapeHtml(sample.asset_urls.mask)}" target="_blank">Open mask</a>
      </article>
    `;
  }).join("");
  results.querySelectorAll("[data-review-job]").forEach((button) => {
    button.addEventListener("click", () => mountReviewRecord(button.dataset.reviewJob, button.dataset.reviewLabel, button.dataset.reviewKey, button.dataset.reviewPath));
  });
}

async function runFluorescencePrediction() {
  if (state.fluorescence.running || !state.fluorescence.selected.size) {
    return;
  }
  const button = document.getElementById("fluorescence-run");
  const progress = document.getElementById("fluorescence-progress");
  const report = document.getElementById("fluorescence-report");
  state.fluorescence.running = true;
  button.disabled = true;
  button.textContent = "Running...";
  progress.textContent = `Running segmentation and viability prediction for ${state.fluorescence.selected.size} image(s). Do not submit another GPU job.`;
  const progressBar = document.getElementById("fluorescence-progress-bar");
  if (progressBar) {
    progressBar.hidden = false;
    progressBar.value = 0;
  }
  report.innerHTML = "";
  try {
    state.fluorescence.progressTimer = setInterval(async () => {
      try {
        const status = await fetchJson("/api/fluorescence/status");
        const current = status.progress;
        if (!current) return;
        if (progressBar) progressBar.value = Number(current.percent || 0);
        const currentFile = current.current_file ? ` Current: ${current.current_file}.` : "";
        progress.textContent = `${current.completed}/${current.total} completed (${Number(current.percent || 0).toFixed(1)}%); ${current.succeeded} succeeded, ${current.failed} failed.${currentFile} Elapsed: ${Number(current.elapsed_seconds || 0).toFixed(0)} s.`;
      } catch (_) {
        // Keep the last visible progress while a poll is temporarily unavailable.
      }
    }, 1000);
    const data = await postJson("/api/fluorescence/run", {
      paths: Array.from(state.fluorescence.selected),
      order: document.getElementById("fluorescence-order").value,
      instruction: document.getElementById("fluorescence-instruction").value,
      language: document.getElementById("fluorescence-language").value,
    });
    renderFluorescenceResults(data);
    if (progressBar) progressBar.value = 100;
    progress.textContent = `Completed ${data.summary.succeeded} image(s); ${data.summary.failed} failed.`;
    const reportResponse = await fetch(data.report_url);
    report.innerHTML = reportResponse.ok ? renderMarkdown(await reportResponse.text()) : "";
    const recordId = `viability:${data.job_id}`;
    if (!state.results.records.some((item) => item.id === recordId)) {
      state.results.records.unshift({
        id: recordId,
        kind: "viability",
        title: `Viability · ${data.job_id}`,
        status: data.status || "succeeded",
        created_at: Date.now() / 1000,
        sample_count: data.samples?.length || 0,
        payload: data,
      });
      state.results.selectedId = recordId;
      renderSessionResultsSummary();
      renderResultsHistory();
    }
    setActiveStage("results", "viability-results");
  } catch (err) {
    progress.textContent = `Prediction failed: ${err.message}`;
  } finally {
    if (state.fluorescence.progressTimer) {
      clearInterval(state.fluorescence.progressTimer);
      state.fluorescence.progressTimer = null;
    }
    state.fluorescence.running = false;
    button.disabled = false;
    button.textContent = "Run prediction";
  }
}

async function initFluorescence() {
  const select = document.getElementById("fluorescence-dataset");
  if (!select) {
    return;
  }
  const statusEl = document.getElementById("fluorescence-status");
  select.innerHTML = state.datasets.map((dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`).join("");
  try {
    const status = await fetchJson("/api/fluorescence/status");
    statusEl.textContent = status.ready ? "ready" : "not ready";
  } catch (err) {
    statusEl.textContent = "status unavailable";
    statusEl.title = err.message;
  }
  select.addEventListener("change", () => loadFluorescenceImages(select.value));
  document.getElementById("fluorescence-select-all")?.addEventListener("click", () => {
    state.fluorescence.selected = new Set(state.fluorescence.images.slice(0, 30).map((item) => item.path));
    renderFluorescenceImages();
  });
  document.getElementById("fluorescence-clear")?.addEventListener("click", () => {
    state.fluorescence.selected.clear();
    renderFluorescenceImages();
  });
  document.getElementById("fluorescence-run")?.addEventListener("click", runFluorescencePrediction);
  if (state.datasets.length) {
    const demo = state.datasets.find((dataset) => dataset.path === "05_Fluorescence_demo");
    select.value = (demo || state.datasets[0]).path;
    await loadFluorescenceImages(select.value);
  }
}

document.querySelectorAll(".stage-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveStage(tab.dataset.stage);
  });
});

document.querySelectorAll(".subtab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveTab(tab.dataset.tab);
  });
});

document.getElementById("clear-session-results")?.addEventListener("click", () => {
  state.results.records = [];
  state.results.selectedId = null;
  state.results.recordedJobIds.clear();
  renderSessionResultsSummary();
  renderResultsHistory();
  showOnlyResultPanel();
  document.getElementById("results-history-detail").textContent = "Select a run to inspect its outputs.";
});

loadDatasets().then(() => {
  initFluorescence();
  initAgentDatasetSelector();
}).catch((err) => {
  document.getElementById("dataset-stats").textContent = err.message;
});

loadSegmentationCategory().catch((err) => {
  const list = document.getElementById("segmentation-group-list");
  if (list) list.textContent = `Unable to load segmentation runs: ${err.message}`;
});
loadCategory("features", "features-list", "features-preview").catch(() => {});
loadCategory("analysis", "analysis-list", "analysis-preview").catch(() => {});
initAgentStudio();
renderSessionResultsSummary();
renderResultsHistory();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/static/sw.js", { scope: "/" });
}
