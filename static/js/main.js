(function() {
  'use strict';

  let charts = {};
  let currentData = null;
  let currentPage = 1;
  let pageSize = 25;
  let sessionId = null;
  let emotionLabels = ['Marah','Takut','Jijik','Sedih','Antisipasi','Percaya','Terkejut','Senang'];
  let emotionColors = ['#ef4444','#f97316','#84cc16','#6366f1','#a855f7','#0ea5e9','#eab308','#10b981'];

  /* ---- DOM References ---- */
  const $ = id => document.getElementById(id);
  const dropZone = $('drop-zone');
  const fileInput = $('file-input');
  const dropContent = $('drop-content');
  const filePreview = $('file-preview');
  const fileName = $('file-name');
  const fileSize = $('file-size');
  const removeBtn = $('remove-file');
  const btnAnalyze = $('btn-analyze');
  const loadingSection = $('loading-section');
  const errorSection = $('error-section');
  const errorMessage = $('error-message');
  const dashboardSection = $('dashboard-section');
  const uploadSection = $('upload-section');
  const loadingStatus = $('loading-status');
  const progressFill = $('progress-fill');
  const summaryText = $('summary-text');
  const summaryDup = $('summary-dup');
  const dupCount = $('dup-count');
  const summaryContent = $('summary-content');
  const wordCloud = $('word-cloud');
  const tableHead = $('table-head');
  const tableBody = $('table-body');
  const tableInfo = $('table-info');
  const tablePage = $('table-page');
  const btnPrev = $('btn-prev');
  const btnNext = $('btn-next');
  const modeBadge = $('mode-badge');
  const modeIndicator = $('mode-indicator');
  const viralBody = $('viral-body');
  const filterSentiment = $('filter-sentiment');
  const filterSearch = $('filter-search');

  let selectedFile = null;

  /* ---- Check API Key Status ---- */
  fetch('/api/key-status')
    .then(r => r.json())
    .then(d => {
      modeBadge.textContent = 'Mode: ' + (d.has_key ? 'DeepSeek AI' : 'Lexicon');
      modeBadge.style.color = d.has_key ? '#10b981' : '#94a3b8';
      modeBadge.style.borderColor = d.has_key ? 'rgba(16, 185, 129, 0.3)' : '#1e293b';
    })
    .catch(() => {});

  /* ---- File Drop/Select Handling ---- */
  dropZone.addEventListener('click', function(e) {
    if (e.target.closest('.btn-ghost') || e.target.closest('.file-preview')) return;
    if (!selectedFile) fileInput.click();
  });
  
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  
  dropZone.addEventListener('dragleave', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  });
  
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });
  
  fileInput.addEventListener('change', function() {
    if (this.files.length > 0) handleFile(this.files[0]);
  });
  
  removeBtn.addEventListener('click', e => {
    e.stopPropagation();
    clearFile();
  });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv','xlsx','xls','html','htm'].includes(ext)) {
      showError('Format file tidak didukung. Gunakan .csv, .xlsx, atau .html.');
      return;
    }
    if (file.size > 50*1024*1024) {
      showError('Ukuran file terlalu besar (maksimum 50MB).');
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

  /* ---- Error/Retry Handlers ---- */
  $('btn-retry').addEventListener('click', () => {
    hideError();
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  $('btn-new-analysis').addEventListener('click', () => {
    dashboardSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  /* ---- Filtering Handlers ---- */
  filterSentiment.addEventListener('change', () => {
    if (currentData) {
      currentPage = 1;
      renderTable(currentData);
    }
  });

  filterSearch.addEventListener('input', () => {
    if (currentData) {
      currentPage = 1;
      renderTable(currentData);
    }
  });

  /* ---- Analysis Pipeline Initiation ---- */
  btnAnalyze.addEventListener('click', () => {
    if (selectedFile) startAnalysis();
  });

  function startAnalysis() {
    const doStemming = $('stemming-toggle').checked;
    const analysisMethod = $('method-select').value;
    uploadSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    hideError();
    setProgress(10, 'Memvalidasi struktur file...');
    activateStep(1);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('stemming', doStemming ? '1' : '0');
    formData.append('analysis_method', analysisMethod);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    
    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        setProgress(10 + Math.round((e.loaded / e.total) * 20), 'Mengunggah dataset...');
      }
    };

    xhr.onload = function() {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const result = JSON.parse(xhr.responseText);
          setProgress(100, 'Analisis selesai! Memuat dashboard...');
          activateStep(4);
          setTimeout(() => {
            loadingSection.classList.add('hidden');
            dashboardSection.classList.remove('hidden');
            renderDashboard(result);
            const scrollOffset = document.querySelector('.summary-bar').offsetTop - 90;
            window.scrollTo({ top: scrollOffset, behavior: 'smooth' });
          }, 600);
        } catch (err) {
          showError('Gagal memproses respons server.');
          loadingSection.classList.add('hidden');
          uploadSection.classList.remove('hidden');
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          showError(err.error || 'Terjadi kesalahan sistem.');
        } catch (e) {
          showError('Kesalahan server (Status: ' + xhr.status + ').');
        }
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
      }
    };

    xhr.onerror = () => {
      showError('Koneksi ke server gagal.');
      loadingSection.classList.add('hidden');
      uploadSection.classList.remove('hidden');
    };

    // Scale progress timers based on selected method
    let cleanTime = 500, scoreTime = 1400, aggTime = 2400;
    if (analysisMethod === 'lexicon') {
      cleanTime = 150; scoreTime = 300; aggTime = 450;
    } else if (analysisMethod === 'hybrid') {
      cleanTime = 450; scoreTime = 1300; aggTime = 2200;
    } else { // full_ai
      cleanTime = 1200; scoreTime = 3500; aggTime = 7000;
    }

    setTimeout(() => { setProgress(40, 'Membersihkan konten & menghapus duplikat...'); activateStep(2); }, cleanTime);
    setTimeout(() => { setProgress(70, 'Menganalisis sentimen teks & klasifikasi emosi...'); activateStep(3); }, scoreTime);
    setTimeout(() => { setProgress(90, 'Melakukan agregasi statistik...'); }, aggTime);
    
    xhr.send(formData);
  }

  function setProgress(percentage, status) {
    progressFill.style.width = Math.min(percentage, 100) + '%';
    if (status) loadingStatus.textContent = status;
  }

  function activateStep(stepNum) {
    document.querySelectorAll('.step').forEach(el => {
      const step = parseInt(el.dataset.step);
      el.classList.toggle('done', step < stepNum);
      el.classList.toggle('active', step === stepNum);
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

  /* ---- Dashboard Redraw ---- */
  function renderDashboard(data) {
    currentData = data;
    sessionId = data.session_id;
    currentPage = 1;

    let modeText = 'Mode: Lexicon Lokal';
    let modeColor = '#94a3b8';
    let modeBorder = '#1e293b';

    if (data.analysis_mode === 'deepseek_ai') {
      modeText = 'Mode: DeepSeek AI (Penuh)';
      modeColor = '#10b981';
      modeBorder = 'rgba(16, 185, 129, 0.3)';
    } else if (data.analysis_mode === 'deepseek_ai_hybrid') {
      modeText = 'Mode: DeepSeek AI (Hibrida)';
      modeColor = '#3b82f6';
      modeBorder = 'rgba(59, 130, 246, 0.3)';
    }

    modeIndicator.textContent = modeText;
    modeIndicator.style.color = modeColor;
    modeIndicator.style.borderColor = modeBorder;

    // Update KPI numbers with animated counting
    animateKPI('kpi-total-post', data.kpis.total_post);
    animateKPI('kpi-accounts', data.kpis.unique_accounts);
    animateKPI('kpi-views', data.kpis.total_views);
    animateKPI('kpi-engagement', data.kpis.total_engagement);
    animateKPI('kpi-pos-pct', data.kpis.positive_pct + '%');
    animateKPI('kpi-neu-pct', data.kpis.neutral_pct + '%');
    animateKPI('kpi-neg-pct', data.kpis.negative_pct + '%');

    // Trend arrows
    const trend = data.kpis.trend || 'stable';
    const tPos = $('trend-pos');
    const tNeg = $('trend-neg');
    tPos.textContent = '';
    tNeg.textContent = '';
    tPos.className = 'kpi-trend';
    tNeg.className = 'kpi-trend';

    if (trend === 'up') {
      tPos.textContent = '↑ Dominan';
      tPos.className = 'kpi-trend up';
    } else if (trend === 'down') {
      tNeg.textContent = '↑ Dominan';
      tNeg.className = 'kpi-trend down';
    }

    summaryText.innerHTML = 'Menganalisis <strong>' + formatNumRaw(data.total_rows) + '</strong> postingan dari <strong>' + formatNumRaw(data.kpis.unique_accounts) + '</strong> akun unik';
    if (data.dup_count > 0) {
      summaryDup.hidden = false;
      dupCount.textContent = formatNumRaw(data.dup_count);
    } else {
      summaryDup.hidden = true;
    }

    renderSentimentDonut(data);
    renderGauge(data);
    renderEmotionRadar(data);
    renderVolumeChart(data);
    renderViralPosts(data);
    renderAccountChart(data);
    renderWordCloud(data);
    renderBigramChart(data);
    renderSummary(data);
    renderTable(data);
  }

  function animateKPI(id, finalVal) {
    const el = $(id);
    if (!el) return;
    const isPercentage = typeof finalVal === 'string' && finalVal.includes('%');
    const target = isPercentage ? parseFloat(finalVal) : parseInt(finalVal);
    
    let current = 0;
    const duration = 800; // Total animation duration in ms
    const frameRate = 30;  // Frames per second
    const totalFrames = Math.max(10, Math.round((duration / 1000) * frameRate));
    const step = target / totalFrames;
    let frame = 0;

    const interval = setInterval(() => {
      frame++;
      current += step;
      if (frame >= totalFrames) {
        clearInterval(interval);
        el.textContent = isPercentage ? target.toFixed(1) + '%' : formatNumRaw(target);
      } else {
        el.textContent = isPercentage ? current.toFixed(1) + '%' : formatNumRaw(Math.round(current));
      }
    }, 1000 / frameRate);
  }

  /* ---- Chart 1: Sentiment Donut ---- */
  function renderSentimentDonut(data) {
    const ctx = document.getElementById('sentiment-donut').getContext('2d');
    if (charts.sentimentDonut) charts.sentimentDonut.destroy();

    charts.sentimentDonut = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Positif', 'Netral', 'Negatif'],
        datasets: [{
          data: [data.kpis.positive, data.kpis.neutral, data.kpis.negative],
          backgroundColor: ['#10b981', '#64748b', '#ef4444'],
          borderWidth: 2,
          borderColor: '#0f1626',
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        cutout: '70%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 18,
              usePointStyle: true,
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 }
            }
          },
          tooltip: {
            backgroundColor: '#0f1626',
            titleColor: '#f8fafc',
            bodyColor: '#94a3b8',
            borderColor: '#1e293b',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: function(ctx) {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pct = total > 0 ? ((ctx.raw / total) * 100).toFixed(1) : 0;
                return ' ' + ctx.label + ': ' + formatNumRaw(ctx.raw) + ' (' + pct + '%)';
              }
            }
          },
          datalabels: {
            display: function(ctx) {
              return ctx.dataset.data[ctx.dataIndex] > 0;
            },
            color: '#f8fafc',
            font: { family: 'Inter', weight: 'bold', size: 12 },
            formatter: function(val, ctx) {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              return total > 0 ? ((val / total) * 100).toFixed(0) + '%' : '';
            }
          }
        }
      },
      plugins: [ChartDataLabels]
    });
  }

  /* ---- Sentiment Index Gauge ---- */
  function renderGauge(data) {
    const index = data.sentiment_index || 0;
    const valEl = $('gauge-value');
    const fillEl = $('gauge-fill');
    
    valEl.textContent = index > 0 ? '+' + index : index;
    const percentage = ((index + 100) / 200) * 100;
    fillEl.style.width = Math.max(0, Math.min(100, percentage)) + '%';
    
    valEl.className = 'gauge-value';
    if (index > 15) {
      valEl.classList.add('pos');
      fillEl.style.background = '#10b981';
    } else if (index < -15) {
      valEl.classList.add('neg');
      fillEl.style.background = '#ef4444';
    } else {
      valEl.classList.add('neu');
      fillEl.style.background = '#64748b';
    }
  }

  /* ---- Chart 2: Emotion Radar ---- */
  function renderEmotionRadar(data) {
    const ctx = document.getElementById('emotion-radar').getContext('2d');
    if (charts.emotionRadar) charts.emotionRadar.destroy();

    let values = emotionLabels.map(() => 0);
    if (data.emotion_distribution) {
      emotionLabels.forEach((emotion, idx) => {
        if (data.emotion_distribution[emotion]) {
          values[idx] = data.emotion_distribution[emotion].percentage || 0;
        }
      });
    }

    charts.emotionRadar = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: emotionLabels,
        datasets: [{
          label: 'Persentase Emosi (%)',
          data: values,
          backgroundColor: 'rgba(59, 130, 246, 0.15)',
          borderColor: '#3b82f6',
          borderWidth: 2,
          pointBackgroundColor: emotionColors,
          pointBorderColor: '#0f1626',
          pointBorderWidth: 2,
          pointRadius: 4.5,
          pointHoverRadius: 6.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f1626',
            borderColor: '#1e293b',
            borderWidth: 1,
            callbacks: {
              label: function(ctx) {
                return ' ' + ctx.label + ': ' + ctx.raw.toFixed(1) + '%';
              }
            }
          }
        },
        scales: {
          r: {
            beginAtZero: true,
            ticks: {
              color: '#64748b',
              backdropColor: 'transparent',
              font: { family: 'Inter', size: 9 }
            },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            angleLines: { color: 'rgba(255, 255, 255, 0.05)' },
            pointLabels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 10, weight: '500' }
            }
          }
        }
      }
    });
  }

  /* ---- Chart 3: Volume Timeline ---- */
  function renderVolumeChart(data) {
    const ctx = document.getElementById('volume-chart').getContext('2d');
    if (charts.volume) charts.volume.destroy();

    if (!data.daily || data.daily.length === 0) {
      ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">Data tren harian tidak tersedia</p>';
      return;
    }

    const labels = data.daily.map(d => {
      const parts = d.date_only ? d.date_only.split('T')[0].split('-') : [];
      return parts.length === 3 ? parts[2] + '/' + parts[1] : d.date_only;
    });

    const spikeDates = {};
    if (data.sentiment_spikes) {
      data.sentiment_spikes.forEach(spike => {
        const shortDate = spike.date.split('-').reverse().slice(0, 2).join('/');
        spikeDates[shortDate] = spike;
      });
    }

    // Dynamic point sizes and colors to highlight spikes
    const positiveRadii = data.daily.map(d => {
      const p = d.date_only ? d.date_only.split('T')[0] : '';
      return (data.sentiment_spikes || []).some(s => s.date === p && s.type === 'Positif') ? 7 : 3.5;
    });
    const positiveHover = positiveRadii.map(r => r + 2.5);
    const positiveColors = data.daily.map(d => {
      const p = d.date_only ? d.date_only.split('T')[0] : '';
      return (data.sentiment_spikes || []).some(s => s.date === p && s.type === 'Positif') ? '#eab308' : '#10b981';
    });

    const negativeRadii = data.daily.map(d => {
      const p = d.date_only ? d.date_only.split('T')[0] : '';
      return (data.sentiment_spikes || []).some(s => s.date === p && s.type === 'Negatif') ? 7 : 3.5;
    });
    const negativeHover = negativeRadii.map(r => r + 2.5);
    const negativeColors = data.daily.map(d => {
      const p = d.date_only ? d.date_only.split('T')[0] : '';
      return (data.sentiment_spikes || []).some(s => s.date === p && s.type === 'Negatif') ? '#eab308' : '#ef4444';
    });

    charts.volume = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Positif',
            data: data.daily.map(d => d.positive || 0),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.04)',
            fill: true,
            tension: 0.25,
            pointRadius: positiveRadii,
            pointHoverRadius: positiveHover,
            pointBackgroundColor: positiveColors,
            pointBorderColor: '#0f1626',
            pointBorderWidth: 1
          },
          {
            label: 'Netral',
            data: data.daily.map(d => d.neutral || 0),
            borderColor: '#64748b',
            backgroundColor: 'rgba(100, 116, 139, 0.02)',
            fill: true,
            tension: 0.25,
            pointRadius: 3.5,
            pointHoverRadius: 6,
            pointBackgroundColor: '#64748b',
            pointBorderColor: '#0f1626',
            pointBorderWidth: 1
          },
          {
            label: 'Negatif',
            data: data.daily.map(d => d.negative || 0),
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.04)',
            fill: true,
            tension: 0.25,
            pointRadius: negativeRadii,
            pointHoverRadius: negativeHover,
            pointBackgroundColor: negativeColors,
            pointBorderColor: '#0f1626',
            pointBorderWidth: 1
          },
          {
            label: 'Engagement (Kanan)',
            data: data.daily.map(d => d.total_engagement || 0),
            borderColor: 'rgba(59, 130, 246, 0.35)',
            backgroundColor: 'transparent',
            tension: 0.25,
            pointRadius: 0,
            yAxisID: 'y1',
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              padding: 16,
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 }
            }
          },
          tooltip: {
            backgroundColor: '#0f1626',
            borderColor: '#1e293b',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              afterBody: function(items) {
                const dateLabel = items[0].label;
                if (spikeDates[dateLabel]) {
                  const s = spikeDates[dateLabel];
                  return '\n⚠️ LONJAKAN SENTIMEN ' + s.type.toUpperCase() + '!\nZ-Score: ' + s.magnitude + 'σ | Rasio: ' + s.pct + '%';
                }
                return '';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } }
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
          },
          y1: {
            position: 'right',
            beginAtZero: true,
            grid: { display: false },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
          }
        }
      }
    });
  }

  /* ---- Viral Posts List ---- */
  function renderViralPosts(data) {
    viralBody.innerHTML = '';
    if (!data.viral_posts || data.viral_posts.length === 0) {
      viralBody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:36px;color:var(--text-muted)">Data postingan viral tidak ditemukan</td></tr>';
      return;
    }
    
    data.viral_posts.slice(0, 10).forEach(post => {
      const sentiment = post.sentiment || 'Neutral';
      const badgeClass = sentiment === 'Positive' ? 'sentiment-positive' : sentiment === 'Negative' ? 'sentiment-negative' : 'sentiment-neutral';
      const label = sentiment === 'Positive' ? 'Positif' : sentiment === 'Negative' ? 'Negatif' : 'Netral';
      
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="akun-cell">${escapeHTML(post.akun)}</td>
        <td class="konten-cell" title="${escapeHTML(post.konten)}">${escapeHTML(post.konten_short || post.konten)}</td>
        <td><span class="sentiment-badge ${badgeClass}">${label}</span></td>
        <td style="font-weight:700;text-align:right;color:var(--text)">${formatNumCompact(post.engagement_score || post.engagement)}</td>
      `;
      viralBody.appendChild(tr);
    });
  }

  /* ---- Chart 4: Account Chart ---- */
  function renderAccountChart(data) {
    const ctx = document.getElementById('account-chart').getContext('2d');
    if (charts.account) charts.account.destroy();

    if (!data.accounts || data.accounts.length === 0) {
      ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:30px">Data aktivitas akun tidak tersedia</p>';
      return;
    }

    const topAccounts = data.accounts.sort((a, b) => b.total_engagement - a.total_engagement).slice(0, 10);
    const labels = topAccounts.map(a => a['X akun'] || '-');
    const engagements = topAccounts.map(a => a.total_engagement || 0);
    
    const colors = topAccounts.map(a => {
      const ds = a.dominant_sentiment || 'Neutral';
      return ds === 'Positive' ? 'rgba(16, 185, 129, 0.65)' : ds === 'Negative' ? 'rgba(239, 68, 68, 0.65)' : 'rgba(100, 116, 139, 0.65)';
    });
    const borderColors = colors.map(c => c.replace('0.65', '1'));

    charts.account = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Engagement',
          data: engagements,
          backgroundColor: colors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f1626',
            borderColor: '#1e293b',
            borderWidth: 1
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 10 } }
          }
        }
      }
    });
  }

  /* ---- Word Cloud Component ---- */
  function renderWordCloud(data) {
    wordCloud.innerHTML = '';
    if (!data.keywords || data.keywords.length === 0) {
      wordCloud.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem">Dataset tidak memiliki keywords</p>';
      return;
    }
    
    const maxCount = data.keywords[0].count;
    data.keywords.forEach((kw, index) => {
      const size = 0.8 + (kw.count / maxCount) * 0.95; // Font sizing: 0.8rem to 1.75rem
      const tagClass = 'cloud-' + (index % 4); // Deterministic gradient colors based on index
      
      const span = document.createElement('span');
      span.className = 'word-tag ' + tagClass;
      span.style.fontSize = size + 'rem';
      span.style.opacity = 0.6 + (kw.count / maxCount) * 0.4;
      span.textContent = kw.word;
      span.title = kw.count + 'x muncul';
      wordCloud.appendChild(span);
    });
  }

  /* ---- Chart 5: Bigrams/Trigrams Chart ---- */
  function renderBigramChart(data) {
    const ctx = document.getElementById('bigram-chart').getContext('2d');
    if (charts.bigram) charts.bigram.destroy();

    const bigrams = data.bigrams || [];
    const trigrams = data.trigrams || [];
    let combined = [];

    bigrams.slice(0, 8).forEach(b => {
      combined.push({ phrase: b.phrase, count: b.count, type: '2-gram' });
    });
    trigrams.slice(0, 5).forEach(t => {
      combined.push({ phrase: t.phrase, count: t.count, type: '3-gram' });
    });
    combined.sort((a, b) => b.count - a.count);
    combined = combined.slice(0, 10);

    if (combined.length === 0) {
      ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:30px">Data frasa kata tidak tersedia</p>';
      return;
    }

    const labels = combined.map(c => c.phrase);
    const counts = combined.map(c => c.count);
    const colors = combined.map(c => c.type === '2-gram' ? 'rgba(59, 130, 246, 0.65)' : 'rgba(168, 85, 247, 0.65)');
    const borderColors = colors.map(c => c.replace('0.65', '1'));

    charts.bigram = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Frekuensi',
          data: counts,
          backgroundColor: colors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f1626',
            borderColor: '#1e293b',
            borderWidth: 1,
            callbacks: {
              label: function(ctx) {
                const item = combined[ctx.dataIndex];
                return ' Terlihat ' + item.count + ' kali (' + item.type + ')';
              }
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 9 } }
          }
        }
      }
    });
  }

  /* ---- Marked.js Summary Parsing ---- */
  function renderSummary(data) {
    if (!data.summary) {
      summaryContent.innerHTML = '<p style="color:var(--text-muted)">Belum ada ringkasan yang dibuat.</p>';
      return;
    }
    summaryContent.innerHTML = marked.parse(data.summary);
  }

  /* ---- Interactive Table Rendering ---- */
  function renderTable(data) {
    let detail = data.detail || [];
    if (detail.length === 0) {
      tableHead.innerHTML = '';
      tableBody.innerHTML = '<tr><td colspan="10" style="text-align:center;padding:40px;color:var(--text-muted)">Dataset tidak memiliki data detail</td></tr>';
      return;
    }

    const filterS = filterSentiment.value;
    const searchQ = filterSearch.value.toLowerCase().trim();

    if (filterS) {
      detail = detail.filter(row => row.sentiment === filterS);
    }
    if (searchQ) {
      detail = detail.filter(row => (row.Konten || '').toLowerCase().includes(searchQ));
    }

    const columnsToDisplay = ['datetime_str', 'X akun', 'Konten', 'sentiment', 'emotion', 'confidence', 'engagement', 'Likes', 'Views', 'Link']
      .filter(col => detail[0].hasOwnProperty(col));

    const friendlyHeaders = {
      'datetime_str': 'Waktu', 'X akun': 'Akun', 'Konten': 'Konten', 'sentiment': 'Sentimen',
      'emotion': 'Emosi', 'confidence': 'Confidence', 'engagement': 'Engagement',
      'Likes': 'Likes', 'Views': 'Views', 'Link': 'Link'
    };

    let headHTML = '<tr>';
    columnsToDisplay.forEach(col => {
      headHTML += '<th>' + (friendlyHeaders[col] || col) + '</th>';
    });
    headHTML += '</tr>';
    tableHead.innerHTML = headHTML;

    // Reset pagination bounds if page exceeds capacity
    const totalPages = Math.ceil(detail.length / pageSize) || 1;
    if (currentPage > totalPages) {
      currentPage = 1;
    }

    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, detail.length);
    const paginatedData = detail.slice(start, end);

    let bodyHTML = '';
    paginatedData.forEach(row => {
      bodyHTML += '<tr>';
      columnsToDisplay.forEach(col => {
        let val = row[col];
        if (col === 'sentiment') {
          const badgeClass = val === 'Positive' ? 'sentiment-positive' : val === 'Negative' ? 'sentiment-negative' : 'sentiment-neutral';
          const label = val === 'Positive' ? 'Positif' : val === 'Negative' ? 'Negatif' : 'Netral';
          bodyHTML += '<td><span class="sentiment-badge ' + badgeClass + '">' + label + '</span></td>';
        } else if (col === 'emotion') {
          const emotionLower = (val || 'Netral').toLowerCase();
          const emotionLabel = val || 'Netral';
          bodyHTML += '<td><span class="emotion-badge emotion-' + emotionLower + '">' + emotionLabel + '</span></td>';
        } else if (col === 'confidence') {
          bodyHTML += '<td>' + (val != null ? Number(val).toFixed(3) : '-') + '</td>';
        } else if (col === 'Link' && val) {
          bodyHTML += '<td><a href="' + val + '" target="_blank" style="color:var(--primary);text-decoration:none;font-weight:600;font-size:0.8rem">Buka Link</a></td>';
        } else if (col === 'Konten') {
          bodyHTML += '<td style="max-width:280px" title="' + escapeHTML(String(val || '')) + '">' + escapeHTML(String(val || '')) + '</td>';
        } else if (['engagement', 'Likes', 'Views'].includes(col)) {
          bodyHTML += '<td style="text-align:right;font-weight:500">' + (val != null && val !== '' ? formatNumRaw(Number(val)) : '-') + '</td>';
        } else {
          bodyHTML += '<td>' + escapeHTML(String(val || '')) + '</td>';
        }
      });
      bodyHTML += '</tr>';
    });

    tableBody.innerHTML = bodyHTML;
    tableInfo.textContent = 'Menampilkan ' + (detail.length > 0 ? (start + 1) : 0) + '-' + end + ' dari ' + formatNumRaw(detail.length) + ' data';
    tablePage.textContent = 'Hal ' + currentPage + ' / ' + totalPages;
    
    btnPrev.disabled = currentPage <= 1;
    btnNext.disabled = currentPage >= totalPages;
  }

  /* ---- Table Pagination Navigation ---- */
  btnPrev.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      renderTable(currentData);
    }
  });

  btnNext.addEventListener('click', () => {
    if (currentData) {
      const detail = currentData.detail || [];
      const filterS = filterSentiment.value;
      const searchQ = filterSearch.value.toLowerCase().trim();
      let filtered = detail;

      if (filterS) filtered = filtered.filter(row => row.sentiment === filterS);
      if (searchQ) filtered = filtered.filter(row => (row.Konten || '').toLowerCase().includes(searchQ));

      const totalPages = Math.ceil(filtered.length / pageSize);
      if (currentPage < totalPages) {
        currentPage++;
        renderTable(currentData);
      }
    }
  });

  /* ---- Downloads and Exporters ---- */
  $('btn-download-csv').addEventListener('click', () => {
    if (sessionId) window.location.href = '/download/' + sessionId + '/csv';
  });
  
  $('btn-download-json').addEventListener('click', () => {
    if (sessionId) window.location.href = '/download/' + sessionId + '/json';
  });
  
  $('btn-download-summary').addEventListener('click', () => {
    if (sessionId) window.location.href = '/download-summary/' + sessionId;
  });
  
  $('btn-print-summary').addEventListener('click', () => {
    window.print();
  });

  /* ---- Formatter Helpers ---- */
  function formatNumCompact(num) {
    if (num === null || num === undefined || isNaN(num)) return '0';
    num = Number(num);
    if (num >= 1000000) return (num / 1000000).toFixed(1) + ' jt';
    if (num >= 1000) return (num / 1000).toFixed(num % 1000 === 0 ? 0 : 1) + ' rb';
    return num.toString();
  }

  function formatNumRaw(num) {
    if (num === null || num === undefined || isNaN(num)) return '0';
    return Number(num).toLocaleString('id-ID');
  }

  function escapeHTML(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
})();
