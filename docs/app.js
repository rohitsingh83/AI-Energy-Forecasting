/**
 * VOLT - AI Energy Demand Forecasting & SOC Dashboard
 * Interactive Client Engine for GitHub Pages (github.io)
 */

'use strict';

// Global application state
const state = {
  dataset: 'synthetic',
  hours: 24,
  page: 'overview',
  summary: null,
  forecast: null,
  history: null,
  evaluation: null,
  scenario: null,
  requestId: 0
};

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (x, d = 1) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const parseDate = t => new Date(String(t).replace(' ', 'T'));
const formatTime = t => parseDate(t).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
const formatDate = t => parseDate(t).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
const formatMoney = x => '₹' + num(x, 2);

function showToast(msg) {
  let toast = $('toast-msg');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast-msg';
    toast.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#1e293b;color:#f8fafc;border:1px solid #10b981;padding:12px 20px;border-radius:8px;font-size:12px;z-index:999;box-shadow:0 10px 25px rgba(0,0,0,0.5);';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.display = 'block';
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.style.display = 'none'; }, 5000);
}

// --------------------------------------------------------------------------
// Offline Simulation Engine (Mirrors Python backend logic)
// --------------------------------------------------------------------------
function calculateOfflineScenario(runKey, overrides) {
  const datasetSource = SNAPSHOT.datasets[runKey];
  const rows = datasetSource.forecast.predictions.slice(0, 24);
  const oldSim = datasetSource.summary.simulation;
  const s = { ...oldSim.settings, ...overrides };

  if (s.offpeak_rate < 0 || s.offpeak_rate > s.standard_rate || s.standard_rate > s.peak_rate || s.peak_rate > 1000) {
    throw new Error('Rates must satisfy 0 ≤ off-peak ≤ standard ≤ peak ≤ 1000.');
  }
  if (s.flexible_fraction < 0 || s.flexible_fraction > 0.5) {
    throw new Error('Flexible share must be 0–50%.');
  }

  const base = rows.map(r => r.predicted_kw);
  const maxKw = Math.max(...base);
  const isPeak = rows.map(r => s.peak_hours.includes(parseDate(r.Datetime).getHours()));
  const isOff = rows.map(r => s.offpeak_hours.includes(parseDate(r.Datetime).getHours()));

  const requested = base.reduce((acc, v, i) => acc + (isPeak[i] ? v * s.flexible_fraction : 0), 0);
  const headroom = base.map((v, i) => isOff[i] ? Math.max(0, maxKw - v) : 0);
  const capacity = headroom.reduce((a, b) => a + b, 0);
  const moved = Math.min(requested, capacity);

  const shifted = base.map((v, i) => {
    const deduct = (isPeak[i] && requested) ? (v * s.flexible_fraction * moved / requested) : 0;
    const add = (isOff[i] && capacity) ? (moved * headroom[i] / capacity) : 0;
    return v - deduct + add;
  });

  const rates = base.map((v, i) => isPeak[i] ? s.peak_rate : isOff[i] ? s.offpeak_rate : s.standard_rate);
  const beforeCost = base.reduce((acc, v, i) => acc + v * rates[i], 0);
  const afterCost = shifted.reduce((acc, v, i) => acc + v * rates[i], 0);
  const totalEnergy = base.reduce((a, b) => a + b, 0);

  const result = {
    ...oldSim,
    settings: s,
    baseline_energy_kwh: totalEnergy,
    shifted_energy_kwh: shifted.reduce((a, b) => a + b, 0),
    requested_shift_kwh: requested,
    moved_energy_kwh: moved,
    baseline_peak_kw: maxKw,
    shifted_peak_kw: Math.max(...shifted),
    baseline_cost_inr: beforeCost,
    shifted_cost_inr: afterCost,
    hypothetical_savings_inr: beforeCost - afterCost,
    hypothetical_savings_pct: beforeCost ? ((beforeCost - afterCost) / beforeCost * 100) : 0
  };

  return {
    dataset: runKey,
    result,
    hourly: rows.map((r, i) => ({
      ...r,
      baseline_kw: base[i],
      shifted_kw: shifted[i],
      rate_inr_per_kwh: rates[i],
      baseline_cost_inr: base[i] * rates[i],
      shifted_cost_inr: shifted[i] * rates[i]
    }))
  };
}

// --------------------------------------------------------------------------
// API Handler (Handles SNAPSHOT data for standalone GitHub Pages)
// --------------------------------------------------------------------------
async function fetchEndpoint(url, body) {
  if (typeof SNAPSHOT !== 'undefined' && SNAPSHOT) {
    const u = new URL(url, 'https://snapshot.local');
    const runKey = body?.dataset || u.searchParams.get('dataset') || state.dataset;
    if (u.pathname === '/api/runs') return SNAPSHOT.runs;

    const d = SNAPSHOT.datasets[runKey];
    if (!d) throw new Error('Requested dataset not found in bundle.');

    if (u.pathname === '/api/summary') return d.summary;
    if (u.pathname === '/api/history') return d.history;
    if (u.pathname === '/api/evaluation') return d.evaluation;
    if (u.pathname === '/api/simulate') return calculateOfflineScenario(runKey, body.settings || {});
    if (u.pathname === '/api/forecast') {
      const h = Number(u.searchParams.get('hours') || 24);
      const predictions = d.forecast.predictions.slice(0, h);
      const peakRow = predictions.reduce((a, b) => a.predicted_kw >= b.predicted_kw ? a : b);
      return {
        ...d.forecast,
        hours: h,
        predictions,
        total_kwh: predictions.reduce((a, b) => a + b.predicted_kwh, 0),
        peak_kw: peakRow.predicted_kw,
        peak_time: peakRow.Datetime
      };
    }
    throw new Error('Endpoint not supported in static mode.');
  }

  // Live Flask server fallback
  const res = await fetch(url, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.message || 'API call failed');
  return json;
}

// --------------------------------------------------------------------------
// SVG Chart Drawing Utility
// --------------------------------------------------------------------------
function drawSvgChart(elementId, seriesList, options = {}) {
  const container = $(elementId);
  if (!container) return;

  const times = options.times || seriesList[0].values.map((_, i) => i);
  const N = times.length;
  const W = 880;
  const H = options.small ? 240 : 320;
  const L = 50, R = 20, T = 25, B = 45;
  const iw = W - L - R, ih = H - T - B;

  let allVals = seriesList.flatMap(s => s.values).filter(v => v != null && Number.isFinite(Number(v)));
  if (options.band) {
    allVals = allVals.concat(
      options.band.lower.filter(v => v != null),
      options.band.upper.filter(v => v != null)
    );
  }

  if (!allVals.length) {
    container.innerHTML = '<div style="padding:40px;text-align:center;color:#64748b;">No data available</div>';
    return;
  }

  const maxVal = Math.max(...allVals) * 1.12 || 1;
  const minVal = 0;
  const getX = i => L + (i / Math.max(1, N - 1)) * iw;
  const getY = v => T + ih - (v - minVal) / (maxVal - minVal) * ih;

  let svg = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(options.label || 'Chart')}">
      <defs>
        <linearGradient id="grad-${elementId}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      <text x="${L}" y="14" font-size="10" fill="#64748b" font-weight="600">${esc(options.unit || 'POWER (kW)')}</text>
  `;

  // Horizontal gridlines & Y-axis labels
  const gridSteps = 5;
  for (let j = 0; j < gridSteps; j++) {
    const val = maxVal * j / (gridSteps - 1);
    const yPos = getY(val);
    svg += `
      <line x1="${L}" y1="${yPos}" x2="${W - R}" y2="${yPos}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3 4"/>
      <text x="${L - 8}" y="${yPos + 3}" font-size="10" fill="#64748b" text-anchor="end">${num(val, maxVal < 10 ? 2 : 0)}</text>
    `;
  }

  // Shaded Empirical Uncertainty Band
  if (options.band) {
    const upperPts = options.band.upper.map((v, i) => v == null ? null : [getX(i), getY(v)]).filter(Boolean);
    const lowerPts = options.band.lower.map((v, i) => v == null ? null : [getX(i), getY(v)]).filter(Boolean);
    if (upperPts.length && lowerPts.length) {
      const polygonCoords = upperPts.concat(lowerPts.reverse());
      svg += `<polygon points="${polygonCoords.map(p => p.join(',')).join(' ')}" fill="rgba(16, 185, 129, 0.12)" stroke="rgba(16, 185, 129, 0.25)" stroke-width="1"/>`;
    }
  }

  // Draw series lines
  seriesList.forEach(s => {
    let d = '', validPoints = [];
    s.values.forEach((v, i) => {
      if (v == null || !Number.isFinite(Number(v))) {
        validPoints.push(null);
        return;
      }
      validPoints.push([getX(i), getY(v)]);
    });

    let hasStarted = false;
    validPoints.forEach(pt => {
      if (!pt) { hasStarted = false; return; }
      d += (hasStarted ? 'L' : 'M') + pt[0].toFixed(2) + ',' + pt[1].toFixed(2) + ' ';
      hasStarted = true;
    });

    if (s.fill) {
      const nonNull = validPoints.filter(Boolean);
      if (nonNull.length) {
        svg += `<path d="M${nonNull[0][0]},${T + ih} ${nonNull.map(p => 'L' + p.join(',')).join(' ')} L${nonNull[nonNull.length - 1][0]},${T + ih}Z" fill="url(#grad-${elementId})"/>`;
      }
    }

    svg += `<path d="${d}" stroke="${s.color || '#10b981'}" stroke-width="${s.width || 2}" stroke-linecap="round" stroke-linejoin="round" fill="none" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>`;
  });

  // Cutoff indicator line
  if (options.cutoff != null) {
    const cx = getX(options.cutoff);
    svg += `
      <line x1="${cx}" y1="${T}" x2="${cx}" y2="${T + ih}" stroke="#f59e0b" stroke-dasharray="4 4" stroke-width="1.5"/>
      <rect x="${Math.min(cx - 40, W - 95)}" y="${T + 4}" width="80" height="18" rx="4" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
      <text x="${Math.min(cx, W - 55)}" y="${T + 16}" text-anchor="middle" fill="#f59e0b" font-size="9" font-weight="700">FORECAST START</text>
    `;
  }

  // X-axis timestamps
  const xTicks = options.small ? 4 : 6;
  for (let j = 0; j < xTicks; j++) {
    const idx = Math.round(j * (N - 1) / (xTicks - 1));
    let label = '';
    if (options.numeric) {
      label = `+${times[idx]} h`;
    } else {
      label = N > 96 ? formatDate(times[idx]).slice(0, 6) : formatTime(times[idx]);
    }
    svg += `<text x="${getX(idx)}" y="${H - 14}" text-anchor="${j === 0 ? 'start' : j === xTicks - 1 ? 'end' : 'middle'}" fill="#64748b" font-size="10">${esc(label)}</text>`;
  }

  svg += '</svg><div class="chart-tooltip"></div>';
  container.innerHTML = svg;

  // Tooltip interaction
  const svgEl = container.querySelector('svg');
  const tooltip = container.querySelector('.chart-tooltip');

  svgEl.addEventListener('pointermove', e => {
    const rect = svgEl.getBoundingClientRect();
    const cursorX = (e.clientX - rect.left) / rect.width * W;
    const closestIdx = Math.max(0, Math.min(N - 1, Math.round((cursorX - L) / iw * (N - 1))));
    
    const activeValues = seriesList.filter(s => s.values[closestIdx] != null).map(s => {
      return `<span style="color:${s.color || '#10b981'}">●</span> ${esc(s.name)}: <strong>${num(s.values[closestIdx], 2)} kW</strong>`;
    });

    if (!activeValues.length) return;

    let timeHeader = options.numeric 
      ? `Lead time: +${times[closestIdx]} hours` 
      : `${formatDate(times[closestIdx])} · ${formatTime(times[closestIdx])}`;

    tooltip.innerHTML = `<div style="font-size:10px;color:#94a3b8;margin-bottom:4px">${esc(timeHeader)}</div>` + activeValues.join('<br>');
    tooltip.style.display = 'block';
    
    const tipWidth = 190;
    const leftPos = Math.min(e.clientX - rect.left + 12, rect.width - tipWidth);
    tooltip.style.left = Math.max(8, leftPos) + 'px';
    tooltip.style.top = Math.max(10, e.clientY - rect.top - 70) + 'px';
  });

  svgEl.addEventListener('pointerleave', () => {
    tooltip.style.display = 'none';
  });
}

// --------------------------------------------------------------------------
// Navigation & Views
// --------------------------------------------------------------------------
function switchTab(pageId) {
  state.page = pageId;
  document.querySelectorAll('.tab-content').forEach(el => el.classList.toggle('active', el.id === 'tab-' + pageId));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === pageId));
  
  try {
    if (location.hash !== '#' + pageId) {
      history.replaceState(null, '', '#' + pageId);
    }
  } catch (_) {}
}

// --------------------------------------------------------------------------
// Data Rendering
// --------------------------------------------------------------------------
function renderOverview() {
  const s = state.summary;
  const m = s.metrics;
  const q = s.quality;
  const f = state.forecast;
  const sim = state.scenario?.result || s.simulation;

  // Banner Provenance
  $('dataset-pill').textContent = m.source.source_type === 'synthetic' ? 'SIMULATED BUILDING' : 'MEASURED HOUSEHOLD';
  $('dataset-provenance').textContent = `${num(q.hourly_rows, 0)} hourly meter readings · data through ${formatDate(q.end)} · chronological evaluation`;

  // Top KPIs
  $('kpi-kwh').innerHTML = num(f.total_kwh, 0) + '<small>kWh</small>';
  $('kpi-peak').innerHTML = num(f.peak_kw, 2) + '<small>kW</small>';
  $('kpi-peak-at').textContent = `${formatTime(f.peak_time)} (${formatDate(f.peak_time).slice(0, 6)})`;
  $('kpi-rmse').innerHTML = num(m.test_day_ahead.rmse_kw, 3) + '<small>kW</small>';
  $('kpi-saving').innerHTML = '₹' + num(sim.hypothetical_savings_inr, 2);
  $('kpi-saving-pct').textContent = `${num(sim.hypothetical_savings_pct, 1)}% modeled reduction`;

  // Planning Brief
  $('brief-peak-time').textContent = formatTime(f.peak_time);
  $('brief-peak-date').textContent = formatDate(f.peak_time);
  $('brief-champion').textContent = m.selected_model_label;

  // Overview 48h Context Chart
  const histRows = state.history.data.slice(-48);
  const futureRows = f.predictions;
  const allTimes = histRows.map(r => r.Datetime).concat(futureRows.map(r => r.Datetime));
  const cutoffIndex = histRows.length;

  drawSvgChart('overview-chart', [
    { name: 'Historical Meter', values: histRows.map(r => r.Energy_kW).concat(futureRows.map(() => null)), color: '#64748b' },
    { name: 'Model Forecast', values: histRows.map(() => null).concat(futureRows.map(r => r.predicted_kw)), color: '#10b981', fill: true }
  ], {
    times: allTimes,
    cutoff: cutoffIndex,
    band: {
      lower: histRows.map(() => null).concat(futureRows.map(r => r.lower_kw)),
      upper: histRows.map(() => null).concat(futureRows.map(r => r.upper_kw))
    },
    label: 'Historical readings and recursive future forecast'
  });

  // Forecast Explorer Chart
  drawSvgChart('forecast-chart', [
    { name: 'Future Demand', values: futureRows.map(r => r.predicted_kw), color: '#06b6d4', fill: true }
  ], {
    times: futureRows.map(r => r.Datetime),
    band: {
      lower: futureRows.map(r => r.lower_kw),
      upper: futureRows.map(r => r.upper_kw)
    },
    label: 'Future hourly forecast with uncertainty band'
  });

  $('forecast-window-label').textContent = `${formatDate(futureRows[0].Datetime)} ${formatTime(futureRows[0].Datetime)} → ${formatDate(futureRows[futureRows.length - 1].Datetime)} ${formatTime(futureRows[futureRows.length - 1].Datetime)}`;

  // Forecast Table (First 24-48 rows)
  const displaySlice = futureRows.slice(0, 48);
  $('forecast-table-body').innerHTML = displaySlice.map(r => `
    <tr>
      <td><code>${esc(r.Datetime)}</code></td>
      <td><strong>+${r.horizon} h</strong></td>
      <td><span style="color:#10b981;font-weight:700">${num(r.predicted_kw, 3)}</span></td>
      <td>${num(r.predicted_kwh, 3)}</td>
      <td>${r.lower_kw != null ? num(r.lower_kw, 3) : '—'}</td>
      <td>${r.upper_kw != null ? num(r.upper_kw, 3) : '—'}</td>
    </tr>
  `).join('');

  // Horizon buttons active state
  document.querySelectorAll('[data-hours]').forEach(b => {
    b.classList.toggle('active', Number(b.dataset.hours) === state.hours);
  });
}

function renderEvidence() {
  const e = state.evaluation;
  const m = state.summary.metrics;

  $('ev-rmse-val').innerHTML = num(m.test_day_ahead.rmse_kw, 3) + '<small>kW</small>';
  $('ev-mae-val').innerHTML = num(m.test_day_ahead.mae_kw, 3) + '<small>kW</small>';
  $('ev-r2-val').textContent = num(m.test_day_ahead.r2, 3);
  $('ev-improv-val').textContent = num(m.day_ahead_rmse_improvement_vs_weekly_baseline_pct, 1) + '%';

  // Candidate Comparison Table
  $('comparison-table-body').innerHTML = e.comparison.map(r => `
    <tr class="${r.selected ? 'winner-row' : ''}">
      <td>
        <strong>${esc(r.label)}</strong>
        ${r.selected ? '<span class="winner-tag">CHAMPION</span>' : ''}
      </td>
      <td><strong>${num(r.validation_day_ahead_rmse_kw, 3)}</strong></td>
      <td>${num(r.test_day_ahead_rmse_kw, 3)}</td>
      <td>${num(r.test_one_step_rmse_kw, 3)}</td>
      <td>${num(r.test_day_ahead_r2, 3)}</td>
    </tr>
  `).join('');

  // One-step test replay chart
  drawSvgChart('one-step-chart', [
    { name: 'Observed', values: e.one_step.map(r => r.actual_kw), color: '#64748b' },
    { name: 'Predicted', values: e.one_step.map(r => r.predicted_kw), color: '#10b981' }
  ], {
    times: e.one_step.map(r => r.Datetime),
    small: true,
    label: 'One-step held-out test predictions'
  });

  // Day-ahead backtest example chart
  drawSvgChart('day-ahead-chart', [
    { name: 'Observed', values: e.day_ahead_example.map(r => r.actual_kw), color: '#64748b' },
    { name: '24h Forecast', values: e.day_ahead_example.map(r => r.predicted_kw), color: '#06b6d4' }
  ], {
    times: e.day_ahead_example.map(r => r.Datetime),
    small: true,
    label: '24-hour day-ahead backtest origin'
  });

  // Error by hours ahead chart
  drawSvgChart('horizon-chart', [
    { name: 'Test RMSE', values: e.horizon_metrics.map(r => r.rmse_kw), color: '#f59e0b', fill: true }
  ], {
    times: e.horizon_metrics.map(r => r.horizon),
    numeric: true,
    unit: 'ERROR (kW)',
    small: true,
    label: 'Forecast error by lead time'
  });
}

function renderScenario() {
  const d = state.scenario;
  const r = d.result;

  $('sb-base-cost').textContent = formatMoney(r.baseline_cost_inr);
  $('sb-shifted-cost').textContent = formatMoney(r.shifted_cost_inr);
  $('sb-saving').textContent = formatMoney(r.hypothetical_savings_inr);
  $('sb-saving-pct').textContent = `${num(r.hypothetical_savings_pct, 1)}% modeled savings`;

  drawSvgChart('scenario-chart', [
    { name: 'Baseline Demand', values: d.hourly.map(x => x.baseline_kw), color: '#64748b' },
    { name: 'Shifted Demand', values: d.hourly.map(x => x.shifted_kw), color: '#10b981', fill: true }
  ], {
    times: d.hourly.map(x => x.Datetime),
    label: 'Energy-conserving load shifting scenario'
  });

  $('sb-conserve-note').innerHTML = `
    <strong>Energy Conserved:</strong> ${num(r.baseline_energy_kwh, 2)} kWh before and after (Δ = 0.00 kWh). Moved: <strong>${num(r.moved_energy_kwh, 2)} kWh</strong>.<br>
    <em>Under constant carbon intensity assumptions, modeled emissions reduction is 0 kg.</em>
  `;
}

// --------------------------------------------------------------------------
// Scenario Execution
// --------------------------------------------------------------------------
async function runScenarioSimulation() {
  const btn = $('btn-run-scenario');
  btn.disabled = true;
  try {
    const settings = {
      flexible_fraction: Number($('flex-slider').value) / 100,
      offpeak_rate: Number($('rate-off').value),
      standard_rate: Number($('rate-std').value),
      peak_rate: Number($('rate-peak').value)
    };
    state.scenario = await fetchEndpoint('/api/simulate', { dataset: state.dataset, settings });
    renderScenario();
    showToast('Simulation updated with new parameters');
  } catch (err) {
    showToast('Error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------
// Core Dataset Loader
// --------------------------------------------------------------------------
async function loadDataset(datasetKey) {
  state.dataset = datasetKey;
  const requestId = ++state.requestId;
  
  try {
    const [summary, history, forecast, evaluation, scenario] = await Promise.all([
      fetchEndpoint('/api/summary?dataset=' + datasetKey),
      fetchEndpoint('/api/history?dataset=' + datasetKey + '&hours=168'),
      fetchEndpoint('/api/forecast?dataset=' + datasetKey + '&hours=' + state.hours),
      fetchEndpoint('/api/evaluation?dataset=' + datasetKey),
      fetchEndpoint('/api/simulate', { dataset: datasetKey, settings: {} })
    ]);

    if (requestId !== state.requestId) return;

    state.summary = summary;
    state.history = history;
    state.forecast = forecast;
    state.evaluation = evaluation;
    state.scenario = scenario;

    renderOverview();
    renderEvidence();
    renderScenario();
  } catch (err) {
    showToast('Failed to load dataset: ' + err.message);
  }
}

// --------------------------------------------------------------------------
// Artifact Download Helper
// --------------------------------------------------------------------------
function downloadArtifact(fileType) {
  const d = SNAPSHOT.datasets[state.dataset];
  let text = '';
  let mime = 'application/json';
  let filename = `${state.dataset}_${fileType}`;

  if (fileType === 'forecast_168h.csv') {
    mime = 'text/csv';
    const rows = d.forecast.predictions;
    const keys = Object.keys(rows[0]);
    text = keys.join(',') + '\n' + rows.map(r => keys.map(k => `"${r[k] ?? ''}"`).join(',')).join('\n');
  } else if (fileType === 'model_comparison.csv') {
    mime = 'text/csv';
    const rows = d.evaluation.comparison;
    const keys = Object.keys(rows[0]);
    text = keys.join(',') + '\n' + rows.map(r => keys.map(k => `"${r[k] ?? ''}"`).join(',')).join('\n');
  } else if (fileType === 'metrics.json') {
    text = JSON.stringify(d.summary.metrics, null, 2);
  } else if (fileType === 'simulation.json') {
    text = JSON.stringify(state.scenario.result, null, 2);
  }

  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --------------------------------------------------------------------------
// Screenshot Lightbox Modal
// --------------------------------------------------------------------------
function setupGallery() {
  const modal = $('lightbox-modal');
  const imgEl = $('lightbox-img');
  const titleEl = $('lightbox-title');
  const descEl = $('lightbox-desc');
  const closeBtn = $('lightbox-close');

  document.querySelectorAll('.gallery-card').forEach(card => {
    card.addEventListener('click', () => {
      const src = card.dataset.src;
      const title = card.querySelector('.gallery-title').textContent;
      const desc = card.querySelector('.gallery-desc').textContent;

      imgEl.src = src;
      titleEl.textContent = title;
      descEl.textContent = desc;
      modal.classList.add('active');
    });
  });

  closeBtn.addEventListener('click', () => modal.classList.remove('active'));
  modal.addEventListener('click', e => {
    if (e.target === modal) modal.classList.remove('active');
  });
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  // Navigation tabs
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => switchTab(item.dataset.page));
  });

  // Action jumps
  document.querySelectorAll('[data-goto]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.goto));
  });

  // Horizon switchers
  document.querySelectorAll('[data-hours]').forEach(btn => {
    btn.addEventListener('click', async () => {
      state.hours = Number(btn.dataset.hours);
      state.forecast = await fetchEndpoint('/api/forecast?dataset=' + state.dataset + '&hours=' + state.hours);
      renderOverview();
    });
  });

  // Dataset selector
  $('dataset-dropdown').addEventListener('change', e => {
    loadDataset(e.target.value);
  });

  // Flexible slider input
  $('flex-slider').addEventListener('input', e => {
    $('flex-val-label').textContent = e.target.value + '%';
    runScenarioSimulation();
  });

  // Run scenario button
  $('btn-run-scenario').addEventListener('click', runScenarioSimulation);

  // Setup gallery modal
  setupGallery();

  // Load initial dataset
  const initialHash = location.hash.replace('#', '') || 'overview';
  switchTab(initialHash);
  await loadDataset('synthetic');
});
