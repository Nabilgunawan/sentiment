(function() {
  'use strict';

  let charts = {};
  let currentData = null;
  let currentPage = 1;
  let pageSize = 25;
  let sessionId = null;

  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropContent = document.getElementById('drop-content');
  const filePreview = document.getElementById('file-preview');
  const fileName = document.getElementById('file-name');
  const fileSize = document.getElementById('file-size');
  const removeBtn = document.getElementById('remove-file');
  const btnAnalyze = document.getElementById('btn-analyze');
  const loadingSection = document.getElementById('loading-section');
  const errorSection = document.getElementById('error-section');
  const errorMessage = document.getElementById('error-message');
  const dashboardSection = document.getElementById('dashboard-section');
  const uploadSection = document.getElementById('upload-section');
  const loadingStatus = document.getElementById('loading-status');
  const progressFill = document.getElementById('progress-fill');
  const summaryText = document.getElementById('summary-text');
  const summaryDup = document.getElementById('summary-dup');
  const dupCount = document.getElementById('dup-count');
  const insightContent = document.getElementById('insight-content');
  const keywordCloud = document.getElementById('keyword-cloud');
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  const tableInfo = document.getElementById('table-info');
  const tablePage = document.getElementById('table-page');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');

  let selectedFile = null;

  /* ---- File Selection ---- */
  function openFilePicker() {
    if (selectedFile) return;
    fileInput.click();
  }

  dropZone.addEventListener('click', function(e) {
    if (e.target.closest('.btn-ghost') || e.target.closest('.file-preview')) return;
    openFilePicker();
  });

  dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    this.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', function(e) {
    e.preventDefault();
    this.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    this.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
      handleFile(this.files[0]);
    }
  });

  removeBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    clearFile();
  });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv','xlsx','xls','html','htm'].includes(ext)) {
      showError('Format file tidak didukung. Gunakan file .csv, .xlsx, atau .html.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      showError('File terlalu besar. Maksimum ukuran file adalah 50 MB.');
      return;
    }
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    dropContent.classList.add('hidden');
    filePreview.classList.remove('hidden');
    btnAnalyze.disabled = false;
    hideError();
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    dropContent.classList.remove('hidden');
    filePreview.classList.add('hidden');
    btnAnalyze.disabled = true;
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  /* ---- Retry ---- */
  document.getElementById('btn-retry').addEventListener('click', function() {
    hideError();
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  document.getElementById('btn-new-analysis').addEventListener('click', function() {
    dashboardSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  /* ---- Main Analysis ---- */
  btnAnalyze.addEventListener('click', function() {
    if (!selectedFile) return;
    startAnalysis();
  });

  function startAnalysis() {
    const doStemming = document.getElementById('stemming-toggle').checked;
    uploadSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    hideError();
    setProgress(10, 'Memvalidasi struktur file...');
    activateStep(1);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('stemming', doStemming ? '1' : '0');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);

    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 20);
        setProgress(5 + pct, 'Mengunggah file...');
      }
    };

    xhr.onload = function() {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const result = JSON.parse(xhr.responseText);
          setProgress(100, 'Selesai! Memuat dashboard...');
          activateStep(4);
          setTimeout(function() {
            loadingSection.classList.add('hidden');
            dashboardSection.classList.remove('hidden');
            renderDashboard(result);
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }, 600);
        } catch(e) {
          showError('Gagal memproses respons dari server.');
          loadingSection.classList.add('hidden');
          uploadSection.classList.remove('hidden');
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          showError(err.error || 'Terjadi kesalahan server.');
        } catch(e) {
          showError('Terjadi kesalahan server (status ' + xhr.status + ').');
        }
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
      }
    };

    xhr.onerror = function() {
      showError('Gagal menghubungi server. Pastikan server berjalan.');
      loadingSection.classList.add('hidden');
      uploadSection.classList.remove('hidden');
    };

    setTimeout(function() { setProgress(25, 'Memproses dan membersihkan data...'); activateStep(2); }, 300);
    setTimeout(function() { setProgress(50, 'Menganalisis sentimen...'); }, 800);
    setTimeout(function() { setProgress(75, 'Menghitung agregasi...'); activateStep(3); }, 1500);

    xhr.send(formData);
  }

  function setProgress(pct, status) {
    progressFill.style.width = Math.min(pct, 100) + '%';
    if (status) loadingStatus.textContent = status;
  }

  function activateStep(n) {
    document.querySelectorAll('.step').forEach(function(el) {
      const s = parseInt(el.dataset.step);
      el.classList.toggle('done', s < n);
      el.classList.toggle('active', s === n);
    });
  }

  function hideError() {
    errorSection.classList.add('hidden');
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorSection.classList.remove('hidden');
    if (loadingSection) loadingSection.classList.add('hidden');
  }

  /* ---- Dashboard Rendering ---- */
  function renderDashboard(data) {
    currentData = data;
    sessionId = data.session_id;
    currentPage = 1;

    document.getElementById('kpi-total-post').textContent = formatNum(data.kpis.total_post);
    document.getElementById('kpi-accounts').textContent = formatNum(data.kpis.unique_accounts);
    document.getElementById('kpi-positive').textContent = formatNum(data.kpis.positive);
    document.getElementById('kpi-neutral').textContent = formatNum(data.kpis.neutral);
    document.getElementById('kpi-negative').textContent = formatNum(data.kpis.negative);
    document.getElementById('kpi-engagement').textContent = formatNum(data.kpis.total_engagement);

    summaryText.innerHTML = 'Menganalisis <strong>' + formatNum(data.total_rows) + '</strong> post';
    if (data.dup_count > 0) {
      summaryDup.hidden = false;
      dupCount.textContent = data.dup_count;
    } else {
      summaryDup.hidden = true;
    }

    renderSentimentChart(data);
    renderTrendChart(data);
    renderAccountChart(data);
    renderKeywords(data.keywords);
    renderInsight(data.summary);
    renderTable(data);
  }

  function renderSentimentChart(data) {
    const ctx = document.getElementById('sentiment-chart').getContext('2d');
    if (charts.sentiment) charts.sentiment.destroy();

    charts.sentiment = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Positif', 'Netral', 'Negatif'],
        datasets: [{
          data: [data.kpis.positive, data.kpis.neutral, data.kpis.negative],
          backgroundColor: ['#22c55e', '#eab308', '#ef4444'],
          borderWidth: 0,
          hoverOffset: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        cutout: '65%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { padding: 20, usePointStyle: true, font: { size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const total = ctx.dataset.data.reduce((a,b) => a + b, 0);
                const pct = total > 0 ? (ctx.raw / total * 100).toFixed(1) : 0;
                return ctx.label + ': ' + formatNum(ctx.raw) + ' (' + pct + '%)';
              }
            }
          },
          datalabels: {
            display: function(ctx) {
              return ctx.dataset.data[ctx.dataIndex] > 0;
            },
            color: '#fff',
            font: { weight: 'bold', size: 13 },
            formatter: function(val, ctx) {
              const total = ctx.dataset.data.reduce((a,b) => a + b, 0);
              return total > 0 ? (val / total * 100).toFixed(1) + '%' : '';
            },
          },
        },
      },
      plugins: [ChartDataLabels],
    });
  }

  function renderTrendChart(data) {
    const ctx = document.getElementById('trend-chart').getContext('2d');
    if (charts.trend) charts.trend.destroy();

    if (!data.daily || data.daily.length === 0) {
      document.getElementById('trend-chart').parentElement.innerHTML = '<p class="empty-chart">Data harian tidak tersedia</p>';
      return;
    }

    const labels = data.daily.map(function(d) {
      const parts = d.date_only ? d.date_only.split('T')[0].split('-') : [];
      return parts.length === 3 ? parts[2] + '/' + parts[1] : d.date_only;
    });

    charts.trend = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          { label: 'Positif', data: data.daily.map(function(d) { return d.positive || 0; }), borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: .3, pointRadius: 4, pointHoverRadius: 6 },
          { label: 'Netral', data: data.daily.map(function(d) { return d.neutral || 0; }), borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', fill: true, tension: .3, pointRadius: 4, pointHoverRadius: 6 },
          { label: 'Negatif', data: data.daily.map(function(d) { return d.negative || 0; }), borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: .3, pointRadius: 4, pointHoverRadius: 6 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 11 } } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  function renderAccountChart(data) {
    const ctx = document.getElementById('account-chart').getContext('2d');
    if (charts.account) charts.account.destroy();

    if (!data.accounts || data.accounts.length === 0) {
      document.getElementById('account-chart').parentElement.innerHTML = '<p class="empty-chart">Data akun tidak tersedia</p>';
      return;
    }

    const top = data.accounts.slice(0, 12);
    const labels = top.map(function(a) { return a['X akun'] || '-'; });
    const posts = top.map(function(a) { return a.total_post || 0; });
    const colors = ['#3b82f6','#8b5cf6','#f97316','#22c55e','#ef4444','#eab308','#ec4899','#14b8a6','#6366f1','#f43f5e','#0ea5e9','#84cc16'];

    charts.account = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Total Post',
          data: posts,
          backgroundColor: colors.slice(0, top.length),
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  function renderKeywords(keywords) {
    keywordCloud.innerHTML = '';
    if (!keywords || keywords.length === 0) {
      keywordCloud.innerHTML = '<p style="color:var(--text3);font-size:.9rem;">Belum ada keywords</p>';
      return;
    }
    const maxCount = keywords[0].count;
    keywords.forEach(function(kw) {
      const size = 0.75 + (kw.count / maxCount) * 0.65;
      const opacity = 0.55 + (kw.count / maxCount) * 0.45;
      const tag = document.createElement('span');
      tag.className = 'keyword-tag';
      tag.style.fontSize = (size * 1) + 'rem';
      tag.style.opacity = opacity;
      tag.innerHTML = kw.word + ' <span class="keyword-count">' + kw.count + '</span>';
      keywordCloud.appendChild(tag);
    });
  }

  function renderInsight(text) {
    if (!text) {
      insightContent.innerHTML = '<p style="color:var(--text3)">Belum ada insight tersedia.</p>';
      return;
    }
    insightContent.innerHTML = text.replace(/\n/g, '<br>').replace(/## /g, '<h2>').replace(/# /g, '<h1>').replace(/- /g, '&bull; ').replace(/<h1>/g, '<h1 style="font-size:1.3rem;margin:0 0 12px">').replace(/<h2>/g, '<h2 style="font-size:1.1rem;margin:20px 0 8px;color:var(--text)">');
  }

  /* ---- Table ---- */
  function renderTable(data) {
    if (!data.detail || data.detail.length === 0) {
      tableHead.innerHTML = '';
      tableBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:40px">Tidak ada data untuk ditampilkan</td></tr>';
      return;
    }

    const displayColumns = ['datetime_str', 'X akun', 'Konten', 'sentiment', 'confidence', 'engagement', 'Komentar', 'Repost', 'Likes', 'Views', 'Link'].filter(function(c) {
      return data.detail[0].hasOwnProperty(c);
    });

    const headerLabels = {
      'datetime_str': 'Waktu', 'X akun': 'Akun', 'Konten': 'Konten',
      'sentiment': 'Sentimen', 'confidence': 'Confidence',
      'engagement': 'Engagement', 'Komentar': 'Komentar', 'Repost': 'Repost',
      'Likes': 'Likes', 'Views': 'Views', 'Link': 'Link',
    };

    let html = '<tr>';
    displayColumns.forEach(function(c) {
      html += '<th>' + (headerLabels[c] || c) + '</th>';
    });
    html += '</tr>';
    tableHead.innerHTML = html;

    const totalPages = Math.ceil(data.detail.length / pageSize);
    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, data.detail.length);
    const pageData = data.detail.slice(start, end);

    let bodyHtml = '';
    pageData.forEach(function(row) {
      bodyHtml += '<tr>';
      displayColumns.forEach(function(col) {
        let val = row[col];
        if (col === 'sentiment') {
          const cls = val === 'Positive' ? 'sentiment-positive' : val === 'Negative' ? 'sentiment-negative' : 'sentiment-neutral';
          const label = val === 'Positive' ? 'Positif' : val === 'Negative' ? 'Negatif' : 'Netral';
          bodyHtml += '<td><span class="sentiment-badge ' + cls + '">' + label + '</span></td>';
        } else if (col === 'confidence') {
          bodyHtml += '<td>' + (val !== null && val !== undefined ? Number(val).toFixed(3) : '-') + '</td>';
        } else if (col === 'Link' && val) {
          bodyHtml += '<td><a href="' + val + '" target="_blank" style="color:var(--primary);text-decoration:none;font-size:.8rem">link</a></td>';
        } else if (col === 'Konten') {
          bodyHtml += '<td style="max-width:300px" title="' + escapeHtml(String(val || '')) + '">' + escapeHtml(String(val || '')) + '</td>';
        } else if (['Komentar','Repost','Likes','Views','engagement'].includes(col)) {
          bodyHtml += '<td style="text-align:right">' + (val !== null && val !== undefined && val !== '' ? formatNum(Number(val)) : '-') + '</td>';
        } else {
          bodyHtml += '<td>' + escapeHtml(String(val || '')) + '</td>';
        }
      });
      bodyHtml += '</tr>';
    });
    tableBody.innerHTML = bodyHtml;

    tableInfo.textContent = 'Menampilkan ' + (start + 1) + '-' + end + ' dari ' + formatNum(data.detail.length) + ' data';
    tablePage.textContent = 'Halaman ' + currentPage + ' dari ' + totalPages;
    btnPrev.disabled = currentPage <= 1;
    btnNext.disabled = currentPage >= totalPages;
  }

  btnPrev.addEventListener('click', function() {
    if (currentPage > 1) { currentPage--; renderTable(currentData); }
  });

  btnNext.addEventListener('click', function() {
    if (currentData && currentPage < Math.ceil(currentData.detail.length / pageSize)) {
      currentPage++;
      renderTable(currentData);
    }
  });

  /* ---- Downloads ---- */
  document.getElementById('btn-download-csv').addEventListener('click', function() {
    if (sessionId) window.location.href = '/download/' + sessionId + '/csv';
  });
  document.getElementById('btn-download-json').addEventListener('click', function() {
    if (sessionId) window.location.href = '/download/' + sessionId + '/json';
  });
  document.getElementById('btn-download-summary').addEventListener('click', function() {
    if (sessionId) window.location.href = '/download-summary/' + sessionId;
  });

  /* ---- Helpers ---- */
  function formatNum(n) {
    if (n === null || n === undefined || isNaN(n)) return '0';
    n = Number(n);
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'jt';
    if (n >= 1000) return (n / 1000).toFixed(n % 1000 === 0 ? 0 : 1) + 'rb';
    return n.toString();
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

})();
