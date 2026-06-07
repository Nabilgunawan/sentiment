/**
 * training.js — Model Training Admin Page
 * Handles form submission, progress polling, and evaluation display.
 */
(function () {
  "use strict";

  let pollTimer = null;

  document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("training-form");
    if (form) {
      form.addEventListener("submit", handleSubmit);
    }
    const deployForm = document.getElementById("deploy-form");
    if (deployForm) {
      deployForm.addEventListener("submit", handleDeploy);
    }
  });

  async function handleDeploy(e) {
    e.preventDefault();
    const task = document.getElementById("deploy-task").value;
    const fileInput = document.getElementById("deploy-file");
    const btn = document.getElementById("btn-deploy");
    const statusDiv = document.getElementById("deploy-status");

    if (!fileInput.files.length) {
      statusDiv.style.display = "block";
      statusDiv.style.color = "#ef4444";
      statusDiv.textContent = "Pilih file ZIP terlebih dahulu.";
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    btn.disabled = true;
    btn.textContent = "Mengupload...";
    statusDiv.style.display = "block";
    statusDiv.style.color = "var(--text2)";
    statusDiv.textContent = "Mengupload dan mengekstrak model...";

    try {
      const resp = await fetch(`/train/${task}`, { method: "POST", body: formData });
      const data = await resp.json();

      if (resp.ok) {
        statusDiv.style.color = "#22c55e";
        statusDiv.textContent = `Berhasil! ${data.message} Target: ${data.target_directory}`;
        fileInput.value = "";
      } else {
        statusDiv.style.color = "#ef4444";
        statusDiv.textContent = `Gagal: ${data.error || "Unknown error"}`;
      }
    } catch (err) {
      statusDiv.style.color = "#ef4444";
      statusDiv.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = "Deploy Model";
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);

    const btn = document.getElementById("btn-start-training");
    btn.disabled = true;
    btn.textContent = "Memulai training...";

    try {
      const resp = await fetch("/api/training/start", { method: "POST", body: formData });
      const data = await resp.json();

      if (!resp.ok) {
        alert(data.error || "Gagal memulai training");
        btn.disabled = false;
        btn.textContent = "Mulai Training";
        return;
      }

      // Show progress panel
      document.getElementById("training-progress-panel").style.display = "block";
      document.getElementById("train-status-msg").textContent = data.message;

      // Start polling
      startPolling();
    } catch (err) {
      alert("Error: " + err.message);
      btn.disabled = false;
      btn.textContent = "Mulai Training";
    }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch("/api/training/status");
        const state = await resp.json();
        updateProgress(state);

        if (state.status === "completed" || state.status === "error") {
          clearInterval(pollTimer);
          pollTimer = null;

          const btn = document.getElementById("btn-start-training");
          btn.disabled = false;
          btn.textContent = "Mulai Training";

          if (state.status === "completed") {
            showEvaluation(state);
          } else {
            document.getElementById("train-status-msg").textContent = "Error: " + (state.error || "Unknown error");
            document.getElementById("train-status-msg").style.color = "#ef4444";
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2000);
  }

  function updateProgress(state) {
    const fill = document.getElementById("train-progress-fill");
    fill.style.width = state.progress + "%";

    document.getElementById("train-epoch").textContent = `${state.current_epoch}/${state.total_epochs}`;
    document.getElementById("train-loss").textContent = state.train_loss || "-";
    document.getElementById("val-loss").textContent = state.val_loss || "-";
    document.getElementById("val-accuracy").textContent = state.val_accuracy ? (state.val_accuracy * 100).toFixed(1) + "%" : "-";

    const statusMsg = document.getElementById("train-status-msg");
    if (state.status === "training") {
      statusMsg.textContent = `Training epoch ${state.current_epoch}/${state.total_epochs}...`;
      statusMsg.style.color = "var(--text2)";
    } else if (state.status === "completed") {
      statusMsg.textContent = "✅ Training selesai!";
      statusMsg.style.color = "#22c55e";
    }
  }

  function showEvaluation(state) {
    const panel = document.getElementById("eval-panel");
    const content = document.getElementById("eval-content");
    panel.style.display = "block";

    let html = "";

    if (state.history) {
      const h = state.history;
      html += `<div style="margin-bottom:20px">
        <h4 style="color:var(--text);margin-bottom:8px">Training Summary</h4>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
          <div class="kpi-card"><div class="kpi-label">Best Epoch</div><div class="kpi-value">${h.best_epoch || "-"}</div></div>
          <div class="kpi-card"><div class="kpi-label">Best Val Loss</div><div class="kpi-value">${(h.best_val_loss || 0).toFixed(4)}</div></div>
          <div class="kpi-card"><div class="kpi-label">Best Accuracy</div><div class="kpi-value">${((h.best_val_accuracy || 0) * 100).toFixed(1)}%</div></div>
        </div>
      </div>`;
    }

    if (state.evaluation) {
      const ev = state.evaluation;
      html += `<div style="margin-bottom:16px">
        <h4 style="color:var(--text);margin-bottom:8px">Metrics</h4>
        <table class="data-table" style="font-size:.8rem">
          <tr><td>Accuracy</td><td><strong>${((ev.accuracy || 0) * 100).toFixed(1)}%</strong></td></tr>
          <tr><td>Precision (macro)</td><td>${((ev.precision?.macro || 0) * 100).toFixed(1)}%</td></tr>
          <tr><td>Recall (macro)</td><td>${((ev.recall?.macro || 0) * 100).toFixed(1)}%</td></tr>
          <tr><td>F1 Score (macro)</td><td>${((ev.f1?.macro || 0) * 100).toFixed(1)}%</td></tr>
        </table>
      </div>`;

      if (ev.classification_report) {
        html += `<div style="margin-bottom:16px">
          <h4 style="color:var(--text);margin-bottom:8px">Classification Report</h4>
          <pre style="background:var(--surface2);padding:12px;border-radius:8px;font-size:.7rem;overflow-x:auto;color:var(--text2)">${ev.classification_report}</pre>
        </div>`;
      }
    }

    content.innerHTML = html || "<p>Evaluation data tidak tersedia.</p>";
  }
})();
