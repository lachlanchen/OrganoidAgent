const state = {
  datasets: [],
  files: [],
  activeDataset: null,
};

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
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

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/static/sw.js", { scope: "/" });
}
