const apiBase = window.BIOAGENT_API_BASE || (window.location.port === "8090" ? "" : "http://localhost:8090");

const eyesSelect = document.getElementById("eyes");
const mindSelect = document.getElementById("mind");
const handSelect = document.getElementById("hand");
const taskSelect = document.getElementById("task");
const promptField = document.getElementById("prompt");
const notesField = document.getElementById("notes");
const statusIndicator = document.getElementById("status-indicator");
const jobCount = document.getElementById("job-count");
const jobsContainer = document.getElementById("jobs");

let taskPrompts = {};

function setStatus(text, state) {
  statusIndicator.textContent = text;
  statusIndicator.classList.remove("neutral", "good", "bad");
  statusIndicator.classList.add(state || "neutral");
}

function fillSelect(select, items, defaultId) {
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    select.appendChild(option);
  });
  if (defaultId) {
    select.value = defaultId;
  }
}

async function loadOptions() {
  try {
    const response = await fetch(`${apiBase}/api/options`);
    const data = await response.json();

    taskPrompts = {};
    data.tasks.forEach((task) => {
      taskPrompts[task.id] = task.prompt;
    });

    fillSelect(eyesSelect, data.eyes, data.defaults.eyes);
    fillSelect(mindSelect, data.minds, data.defaults.mind);
    fillSelect(handSelect, data.hands, data.defaults.hand);
    fillSelect(taskSelect, data.tasks, data.defaults.task);

    promptField.value = taskPrompts[data.defaults.task] || "";
    setStatus("Online", "good");
  } catch (error) {
    setStatus("Offline", "bad");
  }
}

function renderJobs(jobs) {
  jobsContainer.innerHTML = "";
  if (!jobs.length) {
    jobsContainer.innerHTML = "<p>No jobs yet. Submit a task to queue one.</p>";
    jobCount.textContent = "0 queued";
    return;
  }

  jobCount.textContent = `${jobs.length} queued`;
  jobs.forEach((job) => {
    const card = document.createElement("div");
    card.className = "job-card";

    const title = document.createElement("h3");
    title.textContent = job.task || "Untitled task";
    card.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "job-meta";
    meta.textContent = `${job.eyes} • ${job.mind} • ${job.hand} • ${job.status}`;
    card.appendChild(meta);

    const summary = document.createElement("div");
    summary.textContent = job.summary || "";
    card.appendChild(summary);

    jobsContainer.appendChild(card);
  });
}

async function fetchJobs() {
  try {
    const response = await fetch(`${apiBase}/api/jobs`);
    const data = await response.json();
    renderJobs(data.jobs || []);
  } catch (error) {
    renderJobs([]);
  }
}

async function runJob() {
  const payload = {
    eyes: eyesSelect.value,
    mind: mindSelect.value,
    hand: handSelect.value,
    task: taskSelect.value,
    prompt: promptField.value,
    notes: notesField.value,
  };

  const response = await fetch(`${apiBase}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (response.ok) {
    await fetchJobs();
  }
}

taskSelect.addEventListener("change", () => {
  const nextPrompt = taskPrompts[taskSelect.value] || "";
  if (!promptField.value || promptField.value === taskPrompts[taskSelect.dataset.prev]) {
    promptField.value = nextPrompt;
  }
  taskSelect.dataset.prev = taskSelect.value;
});

document.getElementById("run").addEventListener("click", () => {
  runJob();
});

document.getElementById("refresh").addEventListener("click", () => {
  fetchJobs();
});

loadOptions();
fetchJobs();
