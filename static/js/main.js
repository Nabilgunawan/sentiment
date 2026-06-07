/**
 * main.js — SentimenID Dashboard
 * Advanced Indonesian Social Media Sentiment Analysis Platform
 * 
 * Features:
 * - Tab-based dashboard navigation
 * - Chart.js visualizations (sentiment, emotion, sarcasm, topics, trends)
 * - Data table with multi-filter (sentiment, emotion, sarcasm, search)
 * - Export buttons (CSV, JSON, DOCX, XLSX, PDF)
 * - Word cloud rendering
 * - Responsive design
 */
(function () {
  "use strict";

  // ── Color Palette ──────────────────────────────────────────────────────────
  const COLORS = {
    positive: "#22c55e",
    neutral: "#64748b",
    negative: "#ef4444",
    positiveBg: "rgba(34,197,94,0.15)",
    neutralBg: "rgba(100,116,139,0.15)",
    negativeBg: "rgba(239,68,68,0.15)",
    sarcasm: "#f59e0b",
    sarcasmBg: "rgba(245,158,11,0.15)",
  };

  const EMOTION_COLORS = {
    Marah: "#ef4444",
    Senang: "#22c55e",
    Sedih: "#3b82f6",
    Takut: "#a855f7",
    Jijik: "#f97316",
    Terkejut: "#eab308",
    Netral: "#64748b",
    Antisipasi: "#06b6d4",
    Percaya: "#14b8a6",
  };

  // ── State ──────────────────────────────────────────────────────────────────
  let DATA = null;
  let SESSION_ID = null;
  let currentPage = 1;
  const PAGE_SIZE = 25;
  let filteredData = [];
  let charts = {};

  // ── Utilities ──────────────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);
  const show = (el) => el?.classList.remove("hidden");
  const hide = (el) => el?.classList.add("hidden");
  const fmt = (n) => (n || 0).toLocaleString("id-ID");

  function animateNumber(el, target, duration = 800) {
    if (!el) return;
    const start = parseInt(el.textContent.replace(/\D/g, "")) || 0;
    const diff = target - start;
    if (diff === 0) { el.textContent = fmt(target); return; }
    const startTime = performance.now();
    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = fmt(Math.round(start + diff * eased));
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function destroyChart(key) {
    if (charts[key]) { charts[key].destroy(); delete charts[key]; }
  }

  // ── Initialize ─────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", () => {
    initUpload();
    initTabs();
    initFilters();
    initExportButtons();
    checkApiStatus();
  });

  // ── API Status Check ───────────────────────────────────────────────────────
  function checkApiStatus() {
    fetch("/api/key-status")
      .then((r) => r.json())
      .then((d) => {
        const badge = $("#mode-badge");
        if (badge) {
          badge.textContent = d.mode === "indobert_transformer" ? "IndoBERT Transformer" : d.mode;
          badge.title = d.message;
        }
      })
      .catch(() => {});
  }

  // ── File Upload ────────────────────────────────────────────────────────────
  function initUpload() {
    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    const btnAnalyze = $("#btn-analyze");
    const removeBtn = $("#remove-file");
    let selectedFile = null;

    if (!dropZone) return;

    dropZone.addEventListener("click", (e) => {
      if (e.target.id !== "remove-file") fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
      selectedFile = file;
      hide($("#drop-content"));
      show($("#file-preview"));
      $("#file-name").textContent = file.name;
      $("#file-size").textContent = (file.size / 1024).toFixed(1) + " KB";
      btnAnalyze.disabled = false;
    }

    removeBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      selectedFile = null;
      show($("#drop-content"));
      hide($("#file-preview"));
      fileInput.value = "";
      btnAnalyze.disabled = true;
    });

    btnAnalyze?.addEventListener("click", () => {
      if (!selectedFile) return;
      uploadAndAnalyze(selectedFile);
    });

    $("#btn-retry")?.addEventListener("click", () => {
      hide($("#error-section"));
      show($("#upload-section"));
    });

    $("#btn-new-analysis")?.addEventListener("click", () => {
      hide($("#dashboard-section"));
      show($("#upload-section"));
      DATA = null;
      SESSION_ID = null;
    });
  }

  function uploadAndAnalyze(file) {
    hide($("#upload-section"));
    hide($("#error-section"));
    show($("#loading-section"));

    // ── Baca status checkbox ──────────────────────────────────────────────────
    const optSentiment  = document.getElementById("opt-sentiment");
    const optEmotion    = document.getElementById("opt-emotion");
    const optSarcasm    = document.getElementById("opt-sarcasm");
    const optTopics     = document.getElementById("opt-topics");
    const optAiSummary  = document.getElementById("opt-ai-summary");

    const options = {
      sentiment: optSentiment  ? optSentiment.checked  : true,
      emotion:   optEmotion    ? optEmotion.checked    : true,
      sarcasm:   optSarcasm    ? optSarcasm.checked    : true,
      topics:    optTopics     ? optTopics.checked     : true,
      ai_summary: optAiSummary ? optAiSummary.checked  : true,
    };

    // ── Steps dinamis sesuai centangan ───────────────────────────────────────
    const steps = ["Memvalidasi struktur file...", "Preprocessing teks..."];
    if (options.sentiment) steps.push("Menganalisis sentimen (IndoBERT)...");
    if (options.emotion)   steps.push("Mendeteksi emosi...");
    if (options.sarcasm)   steps.push("Mendeteksi sarkasme...");
    if (options.topics)    steps.push("Mengekstrak topik...");
    steps.push("Mengagregasi hasil...");

    // Update label progress steps di HTML agar sesuai
    const progressStepEls = $$(".progress-steps .step");
    const stepLabels = ["Validasi", "Preprocessing"];
    if (options.sentiment) stepLabels.push("Sentimen");
    if (options.emotion)   stepLabels.push("Emosi");
    if (options.sarcasm)   stepLabels.push("Sarkasme");
    if (options.topics)    stepLabels.push("Topik");
    stepLabels.push("Agregasi");
    progressStepEls.forEach((el, i) => {
      if (stepLabels[i]) { el.textContent = stepLabels[i]; el.classList.remove("active", "hidden"); }
      else el.classList.add("hidden");
    });

    let stepIdx = 0;
    const progressFill   = $("#progress-fill");
    const loadingStatus  = $("#loading-status");

    const stepTimer = setInterval(() => {
      stepIdx++;
      if (stepIdx < steps.length) {
        loadingStatus.textContent = steps[stepIdx];
        const pct = ((stepIdx + 1) / steps.length) * 90;
        progressFill.style.width = pct + "%";
        progressStepEls.forEach((s, i) => {
          if (!s.classList.contains("hidden")) s.classList.toggle("active", i <= stepIdx);
        });
      }
    }, 2000);

    // ── FormData ─────────────────────────────────────────────────────────────
    const formData = new FormData();
    formData.append("file", file);
    formData.append("options", JSON.stringify(options));

    fetch("/upload", { method: "POST", body: formData })
      .then((r) => {
        clearInterval(stepTimer);
        if (!r.ok) return r.json().then((d) => Promise.reject(d));
        return r.json();
      })
      .then((data) => {
        progressFill.style.width = "100%";
        loadingStatus.textContent = "Selesai!";
        setTimeout(() => {
          hide($("#loading-section"));
          show($("#dashboard-section"));
          DATA = data;
          SESSION_ID = data.session_id;
          applyRanOptions(data.ran_options || options);
          renderDashboard(data);
        }, 500);
      })
      .catch((err) => {
        clearInterval(stepTimer);
        hide($("#loading-section"));
        show($("#error-section"));
        $("#error-message").textContent = err.error || "Terjadi kesalahan.";
      });
  }

  // ── Sembunyikan tab & KPI yang tidak relevan ───────────────────────────────
  function applyRanOptions(ran) {
    // Tab buttons
    const tabMap = {
      emotion:  "emotion",
      sarcasm:  "sarcasm",
      topics:   "topics",
    };
    Object.entries(tabMap).forEach(([opt, tabName]) => {
      const btn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
      if (btn) {
        if (ran[opt] === false) { btn.classList.add("hidden"); }
        else { btn.classList.remove("hidden"); }
      }
    });

    // KPI Sarkasme
    const kpiSarCard = $("#kpi-sarcasm-pct")?.closest(".kpi-card");
    if (kpiSarCard) kpiSarCard.style.display = ran.sarcasm === false ? "none" : "";

    // Emotion radar on overview (hanya sembunyikan canvas jika tidak dijalankan)
    const emotionRadarWrap = document.getElementById("emotion-radar")?.closest(".emotion-radar");
    if (emotionRadarWrap) emotionRadarWrap.style.display = ran.emotion === false ? "none" : "";
  }

  // ── Tab Navigation ─────────────────────────────────────────────────────────
  function initTabs() {
    const tabBtns = $$(".tab-btn");
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;
        tabBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        $$(".tab-content").forEach((tc) => tc.classList.remove("active"));
        const target = $(`#tab-${tab}`);
        if (target) target.classList.add("active");
      });
    });
  }

  // ── Render Dashboard ───────────────────────────────────────────────────────
  function renderDashboard(data) {
    renderSummaryBar(data);
    renderKPIs(data.kpis, data.sarcasm_distribution);
    renderSentimentOverview(data);
    renderVolumeTimeline(data.daily);
    renderKeywordAnalysis(data.keywords, data.bigrams, data.trigrams);
    renderEmotionTab(data);
    renderSarcasmTab(data);
    renderTopicsTab(data);
    renderTrendsTab(data);
    renderInfluencerTab(data);
    renderViralTab(data);
    renderSummaryTab(data);
    renderDataTable(data);
  }

  function renderSummaryBar(data) {
    $("#summary-text").innerHTML = `Menganalisis <strong>${fmt(data.total_rows)}</strong> post`;
    if (data.dup_count > 0) {
      show($("#summary-dup"));
      $("#dup-count").textContent = data.dup_count;
    }
    const modeInd = $("#mode-indicator");
    if (modeInd) {
      modeInd.textContent = "IndoBERT Transformer";
      modeInd.className = "header-badge badge-ai";
    }
  }

  // ── KPI Strip ──────────────────────────────────────────────────────────────
  function renderKPIs(kpis, sarcasmDist) {
    animateNumber($("#kpi-total-post"), kpis.total_post);
    animateNumber($("#kpi-accounts"), kpis.unique_accounts);
    animateNumber($("#kpi-views"), kpis.total_views);
    animateNumber($("#kpi-engagement"), kpis.total_engagement);
    $("#kpi-pos-pct").textContent = kpis.positive_pct + "%";
    $("#kpi-neu-pct").textContent = kpis.neutral_pct + "%";
    $("#kpi-neg-pct").textContent = kpis.negative_pct + "%";
    const sarcasmPct = kpis.sarcasm_pct || (sarcasmDist ? sarcasmDist.sarcasm_pct : 0) || 0;
    $("#kpi-sarcasm-pct").textContent = sarcasmPct + "%";
  }

  // ── Sentiment Overview ─────────────────────────────────────────────────────
  function renderSentimentOverview(data) {
    const kpis = data.kpis;

    // Donut
    destroyChart("sentimentDonut");
    const donutCtx = document.getElementById("sentiment-donut");
    if (donutCtx) {
      charts.sentimentDonut = new Chart(donutCtx, {
        type: "doughnut",
        data: {
          labels: ["Positif", "Netral", "Negatif"],
          datasets: [{
            data: [kpis.positive, kpis.neutral, kpis.negative],
            backgroundColor: [COLORS.positive, COLORS.neutral, COLORS.negative],
            borderWidth: 0,
            hoverOffset: 8,
          }],
        },
        options: {
          cutout: "68%",
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: { position: "bottom", labels: { color: "#94a3b8", font: { family: "Inter", size: 11 }, padding: 16, usePointStyle: true } },
            datalabels: {
              color: "#e2e8f0",
              font: { weight: "bold", size: 12 },
              formatter: (v, ctx) => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                return total ? Math.round((v / total) * 100) + "%" : "";
              },
            },
          },
        },
        plugins: [ChartDataLabels],
      });
    }

    // Gauge
    const gaugeVal = data.sentiment_index || 0;
    const gaugeEl = $("#gauge-value");
    if (gaugeEl) {
      gaugeEl.textContent = gaugeVal > 0 ? "+" + gaugeVal : gaugeVal;
      gaugeEl.style.color = gaugeVal > 20 ? COLORS.positive : gaugeVal < -20 ? COLORS.negative : COLORS.neutral;
    }
    const gaugeFill = $("#gauge-fill");
    if (gaugeFill) {
      const pct = ((gaugeVal + 100) / 200) * 100;
      gaugeFill.style.width = Math.max(2, Math.min(98, pct)) + "%";
      gaugeFill.style.background = gaugeVal > 20 ? COLORS.positive : gaugeVal < -20 ? COLORS.negative : COLORS.neutral;
    }

    // Emotion Radar
    renderEmotionRadar(data.emotion_distribution);
  }

  function renderEmotionRadar(emotionDist) {
    if (!emotionDist) return;
    destroyChart("emotionRadar");
    const ctx = document.getElementById("emotion-radar");
    if (!ctx) return;

    const labels = Object.keys(emotionDist);
    const values = labels.map((l) => {
      const d = emotionDist[l];
      return typeof d === "object" ? d.percentage || 0 : d || 0;
    });
    const colors = labels.map((l) => EMOTION_COLORS[l] || "#64748b");

    charts.emotionRadar = new Chart(ctx, {
      type: "radar",
      data: {
        labels: labels,
        datasets: [{
          label: "Emosi (%)",
          data: values,
          borderColor: "rgba(99,102,241,0.8)",
          backgroundColor: "rgba(99,102,241,0.15)",
          borderWidth: 2,
          pointBackgroundColor: colors,
          pointBorderColor: colors,
          pointRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          r: {
            angleLines: { color: "rgba(148,163,184,0.15)" },
            grid: { color: "rgba(148,163,184,0.1)" },
            pointLabels: { color: "#94a3b8", font: { size: 10, family: "Inter" } },
            ticks: { display: false },
            beginAtZero: true,
          },
        },
        plugins: { legend: { display: false }, datalabels: { display: false } },
      },
    });
  }

  // ── Volume Timeline ────────────────────────────────────────────────────────
  function renderVolumeTimeline(daily) {
    destroyChart("volumeChart");
    const ctx = document.getElementById("volume-chart");
    if (!ctx || !daily?.length) return;

    const labels = daily.map((d) => {
      const dt = new Date(d.date_only);
      return dt.toLocaleDateString("id-ID", { day: "2-digit", month: "short" });
    });

    charts.volumeChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Positif", data: daily.map((d) => d.positive || 0), borderColor: COLORS.positive, backgroundColor: COLORS.positiveBg, fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 },
          { label: "Netral", data: daily.map((d) => d.neutral || 0), borderColor: COLORS.neutral, backgroundColor: COLORS.neutralBg, fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 },
          { label: "Negatif", data: daily.map((d) => d.negative || 0), borderColor: COLORS.negative, backgroundColor: COLORS.negativeBg, fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          y: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" }, beginAtZero: true },
        },
        plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Inter", size: 11 }, usePointStyle: true } }, datalabels: { display: false } },
      },
    });
  }

  // ── Keyword Analysis ───────────────────────────────────────────────────────
  function renderKeywordAnalysis(keywords, bigrams, trigrams) {
    // Word cloud
    const wc = $("#word-cloud");
    if (wc && keywords?.length) {
      wc.innerHTML = "";
      const maxCount = Math.max(...keywords.map((k) => k.count));
      keywords.slice(0, 30).forEach((kw) => {
        const span = document.createElement("span");
        span.className = "cloud-word";
        span.textContent = kw.word;
        const size = 0.7 + (kw.count / maxCount) * 1.5;
        span.style.fontSize = size + "rem";
        span.style.opacity = 0.5 + (kw.count / maxCount) * 0.5;
        const hue = Math.random() * 60 + 200;
        span.style.color = `hsl(${hue}, 70%, 65%)`;
        wc.appendChild(span);
      });
    }

    // Bigram chart
    destroyChart("bigramChart");
    const biCtx = document.getElementById("bigram-chart");
    if (biCtx) {
      const items = [...(bigrams || []).slice(0, 8), ...(trigrams || []).slice(0, 4)];
      charts.bigramChart = new Chart(biCtx, {
        type: "bar",
        data: {
          labels: items.map((i) => i.phrase),
          datasets: [{ data: items.map((i) => i.count), backgroundColor: "rgba(99,102,241,0.6)", borderRadius: 4 }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" } },
            y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          },
          plugins: { legend: { display: false }, datalabels: { display: false } },
        },
      });
    }
  }

  // ── Emotion Tab ────────────────────────────────────────────────────────────
  function renderEmotionTab(data) {
    const dist = data.emotion_distribution;
    if (!dist) return;

    const labels = Object.keys(dist);
    const values = labels.map((l) => typeof dist[l] === "object" ? dist[l].count || 0 : dist[l] || 0);
    const pcts = labels.map((l) => typeof dist[l] === "object" ? dist[l].percentage || 0 : 0);
    const colors = labels.map((l) => EMOTION_COLORS[l] || "#64748b");

    // Donut
    destroyChart("emotionDonut");
    const donutCtx = document.getElementById("emotion-donut");
    if (donutCtx) {
      charts.emotionDonut = new Chart(donutCtx, {
        type: "doughnut",
        data: {
          labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }],
        },
        options: {
          cutout: "60%", responsive: true, maintainAspectRatio: true,
          plugins: {
            legend: { position: "bottom", labels: { color: "#94a3b8", font: { family: "Inter", size: 11 }, usePointStyle: true, padding: 10 } },
            datalabels: {
              color: "#e2e8f0", font: { weight: "bold", size: 11 },
              formatter: (v, ctx) => { const t = ctx.dataset.data.reduce((a, b) => a + b, 0); return t && v > 0 ? Math.round((v / t) * 100) + "%" : ""; },
            },
          },
        },
        plugins: [ChartDataLabels],
      });
    }

    // Bar
    destroyChart("emotionBar");
    const barCtx = document.getElementById("emotion-bar");
    if (barCtx) {
      charts.emotionBar = new Chart(barCtx, {
        type: "bar",
        data: {
          labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6 }],
        },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: true,
          scales: {
            x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" } },
            y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 11 } } },
          },
          plugins: { legend: { display: false }, datalabels: { display: false } },
        },
      });
    }

    // Emotion Trend
    renderEmotionTrend(data.emotion_daily);
  }

  function renderEmotionTrend(emotionDaily) {
    destroyChart("emotionTrend");
    const ctx = document.getElementById("emotion-trend");
    if (!ctx || !emotionDaily?.length) return;

    const dates = emotionDaily.map((d) => {
      const dt = new Date(d.date);
      return dt.toLocaleDateString("id-ID", { day: "2-digit", month: "short" });
    });

    const emotions = Object.keys(EMOTION_COLORS);
    const datasets = emotions.filter((em) => emotionDaily.some((d) => d[em] > 0)).map((em) => ({
      label: em,
      data: emotionDaily.map((d) => d[em] || 0),
      borderColor: EMOTION_COLORS[em],
      backgroundColor: EMOTION_COLORS[em] + "20",
      fill: true,
      tension: 0.3,
      borderWidth: 2,
      pointRadius: 2,
    }));

    charts.emotionTrend = new Chart(ctx, {
      type: "line",
      data: { labels: dates, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          y: { stacked: true, grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" }, beginAtZero: true },
        },
        plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Inter", size: 10 }, usePointStyle: true } }, datalabels: { display: false } },
      },
    });
  }

  // ── Sarcasm Tab ────────────────────────────────────────────────────────────
  function renderSarcasmTab(data) {
    const dist = data.sarcasm_distribution || {};
    const sarcastic = dist.sarcastic || 0;
    const nonSarcastic = dist.non_sarcastic || data.total_rows - sarcastic;

    // Donut
    destroyChart("sarcasmDonut");
    const ctx = document.getElementById("sarcasm-donut");
    if (ctx) {
      charts.sarcasmDonut = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Sarkastik", "Non-Sarkastik"],
          datasets: [{ data: [sarcastic, nonSarcastic], backgroundColor: [COLORS.sarcasm, "#334155"], borderWidth: 0 }],
        },
        options: {
          cutout: "65%", responsive: true, maintainAspectRatio: true,
          plugins: {
            legend: { position: "bottom", labels: { color: "#94a3b8", font: { family: "Inter", size: 11 }, usePointStyle: true } },
            datalabels: {
              color: "#e2e8f0", font: { weight: "bold", size: 13 },
              formatter: (v, ctx) => { const t = ctx.dataset.data.reduce((a, b) => a + b, 0); return t ? Math.round((v / t) * 100) + "%" : ""; },
            },
          },
        },
        plugins: [ChartDataLabels],
      });
    }

    // Sarcastic posts table
    const tbody = $("#sarcasm-body");
    if (tbody && data.detail) {
      tbody.innerHTML = "";
      const sarcasticPosts = data.detail.filter((d) => d.sarcasm === true || d.sarcasm === "True").slice(0, 10);
      if (sarcasticPosts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text3);padding:24px">Tidak ada post sarkastik terdeteksi</td></tr>';
      } else {
        sarcasticPosts.forEach((p) => {
          const konten = (p.Konten || "").substring(0, 100) + ((p.Konten || "").length > 100 ? "..." : "");
          const sentClass = p.sentiment === "Positive" ? "pos" : p.sentiment === "Negative" ? "neg" : "neu";
          tbody.innerHTML += `<tr>
            <td><strong>${p["X akun"] || ""}</strong></td>
            <td>${konten}</td>
            <td><span class="badge ${sentClass}">${p.sentiment || ""}</span></td>
            <td>${((p.sarcasm_confidence || 0) * 100).toFixed(0)}%</td>
          </tr>`;
        });
      }
    }
  }

  // ── Topics Tab ─────────────────────────────────────────────────────────────
  function renderTopicsTab(data) {
    const topics = data.topics || [];

    // Bar chart
    destroyChart("topicBar");
    const ctx = document.getElementById("topic-bar");
    if (ctx && topics.length) {
      const topN = topics.slice(0, 10);
      charts.topicBar = new Chart(ctx, {
        type: "bar",
        data: {
          labels: topN.map((t) => t.name || `Topic ${t.id || ""}`),
          datasets: [{ data: topN.map((t) => t.frequency || t.count || 0), backgroundColor: "rgba(99,102,241,0.6)", borderRadius: 6 }],
        },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: true,
          scales: {
            x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" } },
            y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          },
          plugins: { legend: { display: false }, datalabels: { display: false } },
        },
      });
    }

    // Table
    const tbody = $("#topic-body");
    if (tbody) {
      tbody.innerHTML = "";
      if (!topics.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text3);padding:24px">Topic modeling belum tersedia</td></tr>';
      } else {
        topics.slice(0, 10).forEach((t, i) => {
          const reps = (t.representative_posts || []).slice(0, 2).map((p) => `"${(p || "").substring(0, 60)}..."`).join("<br>");
          tbody.innerHTML += `<tr>
            <td>${i + 1}</td>
            <td><strong>${t.name || "Topic " + (t.id || i)}</strong></td>
            <td>${fmt(t.frequency || t.count || 0)}</td>
            <td style="font-size:.75rem;color:var(--text3)">${reps || "-"}</td>
          </tr>`;
        });
      }
    }
  }

  // ── Trends Tab ─────────────────────────────────────────────────────────────
  function renderTrendsTab(data) {
    // Sentiment trend (default: daily)
    renderSentimentTrend(data.daily, "daily");

    // Trend toggle buttons
    $$(".trend-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".trend-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const period = btn.dataset.period;
        const trendData = period === "weekly" ? data.weekly : period === "monthly" ? data.monthly : data.daily;
        renderSentimentTrend(trendData, period);
      });
    });

    // Spikes table
    const spikeBody = $("#spike-body");
    if (spikeBody) {
      spikeBody.innerHTML = "";
      const spikes = data.sentiment_spikes || [];
      if (!spikes.length) {
        spikeBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:20px">Tidak ada spike terdeteksi</td></tr>';
      } else {
        spikes.forEach((s) => {
          const typeClass = s.type === "Negatif" ? "neg" : "pos";
          spikeBody.innerHTML += `<tr>
            <td>${s.date}</td>
            <td><span class="badge ${typeClass}">${s.type}</span></td>
            <td>${s.magnitude}σ</td>
            <td>${s.count}</td>
            <td>${s.pct}%</td>
          </tr>`;
        });
      }
    }

    // Weighted sentiment chart
    destroyChart("weightedSentiment");
    const wsCtx = document.getElementById("weighted-sentiment-chart");
    if (wsCtx && data.weighted_sentiment) {
      const ws = data.weighted_sentiment;
      charts.weightedSentiment = new Chart(wsCtx, {
        type: "bar",
        data: {
          labels: ["Positif", "Netral", "Negatif"],
          datasets: [{
            label: "Engagement-Weighted %",
            data: [ws.Positive || 0, ws.Neutral || 0, ws.Negative || 0],
            backgroundColor: [COLORS.positive, COLORS.neutral, COLORS.negative],
            borderRadius: 8,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: true,
          scales: {
            y: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8", callback: (v) => v + "%" }, max: 100 },
            x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
          },
          plugins: { legend: { display: false }, datalabels: { color: "#e2e8f0", font: { weight: "bold" }, formatter: (v) => v + "%" } },
        },
        plugins: [ChartDataLabels],
      });
    }
  }

  function renderSentimentTrend(trendData, period) {
    destroyChart("sentimentTrend");
    const ctx = document.getElementById("sentiment-trend");
    if (!ctx || !trendData?.length) return;

    const dateKey = "date_only";
    const labels = trendData.map((d) => {
      const val = d[dateKey] || d.date || d.week || d.month || "";
      try {
        const dt = new Date(val);
        if (period === "monthly") return dt.toLocaleDateString("id-ID", { month: "short", year: "numeric" });
        if (period === "weekly") return dt.toLocaleDateString("id-ID", { day: "2-digit", month: "short" });
        return dt.toLocaleDateString("id-ID", { day: "2-digit", month: "short" });
      } catch { return val; }
    });

    charts.sentimentTrend = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Positif", data: trendData.map((d) => d.positive || 0), borderColor: COLORS.positive, backgroundColor: COLORS.positiveBg, fill: true, tension: 0.3, borderWidth: 2 },
          { label: "Netral", data: trendData.map((d) => d.neutral || 0), borderColor: COLORS.neutral, backgroundColor: COLORS.neutralBg, fill: true, tension: 0.3, borderWidth: 2 },
          { label: "Negatif", data: trendData.map((d) => d.negative || 0), borderColor: COLORS.negative, backgroundColor: COLORS.negativeBg, fill: true, tension: 0.3, borderWidth: 2 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          y: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" }, beginAtZero: true },
        },
        plugins: { legend: { labels: { color: "#94a3b8", usePointStyle: true } }, datalabels: { display: false } },
      },
    });
  }

  // ── Influencer Tab ─────────────────────────────────────────────────────────
  function renderInfluencerTab(data) {
    const influencers = data.influencers || data.accounts || [];

    // Bar chart
    destroyChart("accountChart");
    const ctx = document.getElementById("account-chart");
    if (ctx && influencers.length) {
      const top10 = influencers.slice(0, 10);
      const names = top10.map((a) => a.akun || a["X akun"] || "");
      const engagements = top10.map((a) => a.total_engagement || 0);
      const sentColors = top10.map((a) => {
        const ds = a.dominant_sentiment || "";
        return ds === "Positive" ? COLORS.positive : ds === "Negative" ? COLORS.negative : COLORS.neutral;
      });

      charts.accountChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: names,
          datasets: [{ label: "Engagement", data: engagements, backgroundColor: sentColors, borderRadius: 6 }],
        },
        options: {
          indexAxis: "y", responsive: true, maintainAspectRatio: true,
          scales: {
            x: { grid: { color: "rgba(148,163,184,0.08)" }, ticks: { color: "#94a3b8" } },
            y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 11 } } },
          },
          plugins: { legend: { display: false }, datalabels: { display: false } },
        },
      });
    }

    // Table
    const tbody = $("#influencer-body");
    if (tbody && influencers.length) {
      tbody.innerHTML = "";
      influencers.slice(0, 15).forEach((inf) => {
        const sentClass = (inf.dominant_sentiment || "").toLowerCase().includes("pos") ? "pos" : (inf.dominant_sentiment || "").toLowerCase().includes("neg") ? "neg" : "neu";
        tbody.innerHTML += `<tr>
          <td><strong>${inf.akun || inf["X akun"] || ""}</strong></td>
          <td>${inf.total_post || 0}</td>
          <td>${fmt(inf.total_engagement || 0)}</td>
          <td>${fmt(inf.reach_score || inf.Views || 0)}</td>
          <td><span class="badge ${sentClass}">${inf.dominant_sentiment || "-"}</span></td>
          <td>${inf.dominant_emotion || "-"}</td>
          <td>${((inf.sarcasm_ratio || 0) * 100).toFixed(0)}%</td>
          <td>${(inf.influence_score || 0).toFixed(1)}</td>
        </tr>`;
      });
    }
  }

  // ── Viral Tab ──────────────────────────────────────────────────────────────
  function renderViralTab(data) {
    const tbody = $("#viral-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    const posts = data.viral_posts || [];
    if (!posts.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text3);padding:24px">Tidak ada data</td></tr>';
      return;
    }
    posts.forEach((p) => {
      const konten = (p.konten || "").substring(0, 100) + ((p.konten || "").length > 100 ? "..." : "");
      const sentClass = p.sentiment === "Positive" ? "pos" : p.sentiment === "Negative" ? "neg" : "neu";
      const sarcasmBadge = p.sarcasm ? '<span class="badge sar">Sarkastik</span>' : "";
      tbody.innerHTML += `<tr>
        <td><strong>${p.akun}</strong></td>
        <td>${konten}</td>
        <td><span class="badge ${sentClass}">${p.sentiment}</span></td>
        <td>${p.emotion || "-"}</td>
        <td>${sarcasmBadge}</td>
        <td>
          <div style="font-size:.75rem;line-height:1.6">
            👍 ${fmt(p.likes)} · 🔄 ${fmt(p.repost)} · 💬 ${fmt(p.komentar)}<br>
            <strong>Score: ${(p.engagement_score || 0).toFixed(1)}</strong>
          </div>
        </td>
      </tr>`;
    });
  }

  // ── Summary Tab ────────────────────────────────────────────────────────────
  function renderSummaryTab(data) {
    const summaryEl = $("#summary-content");
    if (summaryEl && data.summary) {
      summaryEl.innerHTML = marked.parse(data.summary);
    }
  }

  // ── Data Table ─────────────────────────────────────────────────────────────
  function renderDataTable(data) {
    if (!data.detail?.length) return;
    filteredData = [...data.detail];
    currentPage = 1;

    // Build header
    const visibleCols = ["X akun", "Konten", "sentiment", "confidence", "emotion", "sarcasm", "engagement", "datetime_str"];
    const colLabels = { "X akun": "Akun", Konten: "Konten", sentiment: "Sentimen", confidence: "Conf", emotion: "Emosi", sarcasm: "Sarkasme", engagement: "Engagement", datetime_str: "Tanggal" };

    const thead = $("#table-head");
    thead.innerHTML = "<tr>" + visibleCols.map((c) => `<th>${colLabels[c] || c}</th>`).join("") + "</tr>";

    renderPage();
  }

  function renderPage() {
    const tbody = $("#table-body");
    if (!tbody) return;

    const start = (currentPage - 1) * PAGE_SIZE;
    const end = start + PAGE_SIZE;
    const page = filteredData.slice(start, end);
    const visibleCols = ["X akun", "Konten", "sentiment", "confidence", "emotion", "sarcasm", "engagement", "datetime_str"];

    tbody.innerHTML = page.map((row) => {
      return "<tr>" + visibleCols.map((c) => {
        let val = row[c] ?? "";
        if (c === "sentiment") {
          const cls = val === "Positive" ? "pos" : val === "Negative" ? "neg" : "neu";
          return `<td><span class="badge ${cls}">${val}</span></td>`;
        }
        if (c === "sarcasm") {
          return `<td>${val === true || val === "True" ? '<span class="badge sar">Ya</span>' : "-"}</td>`;
        }
        if (c === "confidence") return `<td>${((val || 0) * 100).toFixed(0)}%</td>`;
        if (c === "Konten") {
          const short = String(val).substring(0, 80) + (String(val).length > 80 ? "..." : "");
          return `<td style="max-width:300px">${short}</td>`;
        }
        return `<td>${val}</td>`;
      }).join("") + "</tr>";
    }).join("");

    // Footer
    $("#table-info").textContent = `Menampilkan ${start + 1}-${Math.min(end, filteredData.length)} dari ${filteredData.length} data`;
    $("#table-page").textContent = `Hal ${currentPage}`;
    $("#btn-prev").disabled = currentPage === 1;
    $("#btn-next").disabled = end >= filteredData.length;
  }

  // ── Filters ────────────────────────────────────────────────────────────────
  function initFilters() {
    const apply = () => {
      if (!DATA?.detail) return;
      const sentFilter = $("#filter-sentiment")?.value || "";
      const emoFilter = $("#filter-emotion")?.value || "";
      const sarFilter = $("#filter-sarcasm")?.checked || false;
      const search = ($("#filter-search")?.value || "").toLowerCase();

      filteredData = DATA.detail.filter((d) => {
        if (sentFilter && d.sentiment !== sentFilter) return false;
        if (emoFilter && d.emotion !== emoFilter) return false;
        if (sarFilter && d.sarcasm !== true && d.sarcasm !== "True") return false;
        if (search && !(d.Konten || "").toLowerCase().includes(search)) return false;
        return true;
      });

      currentPage = 1;
      renderPage();
    };

    ["filter-sentiment", "filter-emotion"].forEach((id) => {
      $(`#${id}`)?.addEventListener("change", apply);
    });
    $("#filter-sarcasm")?.addEventListener("change", apply);
    $("#filter-search")?.addEventListener("input", apply);

    $("#btn-prev")?.addEventListener("click", () => { if (currentPage > 1) { currentPage--; renderPage(); } });
    $("#btn-next")?.addEventListener("click", () => { if (currentPage * PAGE_SIZE < filteredData.length) { currentPage++; renderPage(); } });
  }

  // ── Export Buttons ─────────────────────────────────────────────────────────
  function initExportButtons() {
    $("#btn-download-csv")?.addEventListener("click", () => {
      if (SESSION_ID) window.location.href = `/download/${SESSION_ID}/csv`;
    });
    $("#btn-download-json")?.addEventListener("click", () => {
      if (SESSION_ID) window.location.href = `/download/${SESSION_ID}/json`;
    });
    $("#btn-download-summary")?.addEventListener("click", () => {
      if (SESSION_ID) window.location.href = `/download-summary/${SESSION_ID}`;
    });
    $("#btn-download-docx")?.addEventListener("click", () => {
      if (SESSION_ID) window.location.href = `/download/${SESSION_ID}/docx`;
    });
    $("#btn-download-xlsx")?.addEventListener("click", () => {
      if (SESSION_ID) window.location.href = `/download/${SESSION_ID}/xlsx`;
    });
    $("#btn-print-summary")?.addEventListener("click", () => window.print());
  }

})();
