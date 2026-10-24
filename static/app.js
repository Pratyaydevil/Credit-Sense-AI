// CreditSense AI - Frontend Controller & Multi-Agent Dashboard

let currentMode = 'FIVE_QUESTION';
let cachedSamples = null;

document.addEventListener('DOMContentLoaded', () => {
  initFormControls();
  initPresetButtons();
  initModal();
  initCopyButton();
  loadSampleProfiles();
  loadBenchmarkMetrics();

  // Trigger initial assessment using default pre-configured benchmark values
  submitAssessment();
});

// -------------------------------------------------------------
// Form Controls & Interactive Sliders
// -------------------------------------------------------------
function initFormControls() {
  const salaryInput = document.getElementById('monthly-salary');
  const salarySlider = document.getElementById('salary-slider');
  const salaryDisplay = document.getElementById('salary-display');

  const debtInput = document.getElementById('outstanding-debt');
  const debtSlider = document.getElementById('debt-slider');
  const debtDisplay = document.getElementById('debt-display');

  const utilInput = document.getElementById('utilization');
  const utilSlider = document.getElementById('utilization-slider');
  const utilDisplay = document.getElementById('utilization-display');

  const delayInput = document.getElementById('delay-days');
  const delaySlider = document.getElementById('delay-slider');
  const delayDisplay = document.getElementById('delay-display');

  const inqInput = document.getElementById('inquiries');
  const inqMinus = document.getElementById('inq-minus');
  const inqPlus = document.getElementById('inq-plus');
  const inqDisplay = document.getElementById('inquiries-display');

  // Salary binding
  syncInputs(salaryInput, salarySlider, val => {
    salaryDisplay.textContent = `$${Number(val).toLocaleString()}`;
    const ann = document.getElementById('annual-income');
    if (ann && !ann.value) ann.placeholder = `$${(Number(val)*12).toLocaleString()} (Auto)`;
  });

  // Debt binding
  syncInputs(debtInput, debtSlider, val => {
    debtDisplay.textContent = `$${Number(val).toLocaleString()}`;
    const emi = document.getElementById('monthly-emi');
    if (emi && !emi.value) emi.placeholder = `$${(Number(val)*0.15).toLocaleString(undefined, {maximumFractionDigits:1})} (Auto)`;
  });

  // Utilization binding
  syncInputs(utilInput, utilSlider, val => {
    const num = parseFloat(val);
    utilDisplay.textContent = `${num.toFixed(1)}%`;
    utilDisplay.className = 'value-chip';
    if (num > 60) utilDisplay.classList.add('critical');
    else if (num > 30) utilDisplay.classList.add('warning');
  });

  // Delay days binding
  syncInputs(delayInput, delaySlider, val => {
    const num = parseFloat(val);
    delayDisplay.textContent = `${num.toFixed(0)} days`;
    delayDisplay.className = 'value-chip';
    if (num > 14) delayDisplay.classList.add('critical');
    else if (num > 5) delayDisplay.classList.add('warning');
  });

  // Inquiries counter
  const updateInqDisplay = () => {
    const v = parseInt(inqInput.value) || 0;
    inqDisplay.textContent = `${v} Inquir${v === 1 ? 'y' : 'ies'}`;
    inqDisplay.className = 'value-chip';
    if (v > 5) inqDisplay.classList.add('critical');
    else if (v > 2) inqDisplay.classList.add('warning');
  };

  inqMinus.addEventListener('click', () => {
    inqInput.value = Math.max(0, (parseInt(inqInput.value) || 0) - 1);
    updateInqDisplay();
  });
  inqPlus.addEventListener('click', () => {
    inqInput.value = Math.min(30, (parseInt(inqInput.value) || 0) + 1);
    updateInqDisplay();
  });
  inqInput.addEventListener('input', updateInqDisplay);

  // Toggle mode buttons
  const mode5q = document.getElementById('mode-5q');
  const modeFull = document.getElementById('mode-full');
  const drawer = document.getElementById('advanced-drawer');

  mode5q.addEventListener('click', () => {
    currentMode = 'FIVE_QUESTION';
    mode5q.classList.add('active');
    modeFull.classList.remove('active');
    if (drawer) drawer.open = false;
  });

  modeFull.addEventListener('click', () => {
    currentMode = 'FULL';
    modeFull.classList.add('active');
    mode5q.classList.remove('active');
    if (drawer) drawer.open = true;
  });

  // Toggle API key visibility
  const toggleKey = document.getElementById('toggle-key-visibility');
  const keyInput = document.getElementById('gemini-api-key');
  toggleKey.addEventListener('click', () => {
    keyInput.type = keyInput.type === 'password' ? 'text' : 'password';
  });

  // Form submit listener
  const form = document.getElementById('applicant-form');
  form.addEventListener('submit', e => {
    e.preventDefault();
    submitAssessment();
  });
}

function syncInputs(numInput, sliderInput, onUpdate) {
  numInput.addEventListener('input', () => {
    sliderInput.value = numInput.value;
    onUpdate(numInput.value);
  });
  sliderInput.addEventListener('input', () => {
    numInput.value = sliderInput.value;
    onUpdate(sliderInput.value);
  });
  onUpdate(numInput.value);
}

// -------------------------------------------------------------
// Presets Loading & Application
// -------------------------------------------------------------
async function loadSampleProfiles() {
  try {
    const res = await fetch('/api/samples');
    if (res.ok) {
      cachedSamples = await res.json();
    }
  } catch (e) {
    console.warn('Could not load sample profiles from server, using built-ins.');
  }
}

function initPresetButtons() {
  const buttons = document.querySelectorAll('.preset-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const presetKey = btn.dataset.preset;
      applyPreset(presetKey);
    });
  });
}

function applyPreset(presetKey) {
  const fallbackPresets = {
    preconfigured: {
      Monthly_Inhand_Salary: 3800,
      Outstanding_Debt: 1650,
      Credit_Utilization_Ratio: 54.0,
      Delay_from_due_date: 12,
      Num_Credit_Inquiries: 4,
      Annual_Income: 48000,
      Total_EMI_per_month: 420
    },
    prime: {
      Monthly_Inhand_Salary: 7500,
      Outstanding_Debt: 420,
      Credit_Utilization_Ratio: 14.5,
      Delay_from_due_date: 0,
      Num_Credit_Inquiries: 1,
      Annual_Income: 90000,
      Total_EMI_per_month: 85
    },
    moderate: {
      Monthly_Inhand_Salary: 4200,
      Outstanding_Debt: 1950,
      Credit_Utilization_Ratio: 42.0,
      Delay_from_due_date: 7,
      Num_Credit_Inquiries: 3,
      Annual_Income: 52000,
      Total_EMI_per_month: 310
    },
    distressed: {
      Monthly_Inhand_Salary: 2100,
      Outstanding_Debt: 4800,
      Credit_Utilization_Ratio: 84.0,
      Delay_from_due_date: 25,
      Num_Credit_Inquiries: 7,
      Annual_Income: 25200,
      Total_EMI_per_month: 620
    }
  };

  const data = (cachedSamples && cachedSamples[presetKey]) ? cachedSamples[presetKey].data : fallbackPresets[presetKey];
  if (!data) return;

  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      el.dispatchEvent(new Event('input'));
    }
  };

  setVal('monthly-salary', data.Monthly_Inhand_Salary);
  setVal('outstanding-debt', data.Outstanding_Debt);
  setVal('utilization', data.Credit_Utilization_Ratio);
  setVal('delay-days', data.Delay_from_due_date);
  setVal('inquiries', data.Num_Credit_Inquiries);

  if (data.Annual_Income) setVal('annual-income', data.Annual_Income);
  if (data.Total_EMI_per_month) setVal('monthly-emi', data.Total_EMI_per_month);

  // Submit assessment immediately for preset selection
  submitAssessment();
}

// -------------------------------------------------------------
// Submit Assessment to Backend (LangGraph Multi-Agent Pipeline)
// -------------------------------------------------------------
async function submitAssessment() {
  const submitBtn = document.getElementById('submit-btn');
  const btnText = submitBtn.querySelector('.btn-text');
  const spinner = submitBtn.querySelector('.spinner');

  // Collect form values
  const salary = parseFloat(document.getElementById('monthly-salary').value) || 0;
  const debt = parseFloat(document.getElementById('outstanding-debt').value) || 0;
  const util = parseFloat(document.getElementById('utilization').value) || 0;
  const delay = parseFloat(document.getElementById('delay-days').value) || 0;
  const inq = parseFloat(document.getElementById('inquiries').value) || 0;

  const annual = parseFloat(document.getElementById('annual-income').value) || (salary * 12.0);
  const emi = parseFloat(document.getElementById('monthly-emi').value) || (debt * 0.15);
  const numLoans = parseInt(document.getElementById('num-loans').value) || 2;
  const histAge = document.getElementById('history-age').value || "5 Years and 6 Months";
  const apiKey = document.getElementById('gemini-api-key').value.trim() || null;

  const payload = {
    Monthly_Inhand_Salary: salary,
    Outstanding_Debt: debt,
    Credit_Utilization_Ratio: util,
    Delay_from_due_date: delay,
    Num_Credit_Inquiries: inq,
    Annual_Income: annual,
    Total_EMI_per_month: emi,
    Num_of_Loan: numLoans,
    Credit_History_Age: histAge,
    mode: currentMode,
    api_key: apiKey
  };

  // Set loading state
  submitBtn.disabled = true;
  btnText.classList.add('hidden');
  spinner.classList.remove('hidden');

  try {
    const res = await fetch('/api/assess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Underwriting assessment failed.');
    }

    const dossier = await res.json();
    renderDossierResults(dossier);
  } catch (err) {
    alert(`Assessment Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    btnText.classList.remove('hidden');
    spinner.classList.add('hidden');
  }
}

// -------------------------------------------------------------
// Render Results onto Dashboard
// -------------------------------------------------------------
function renderDossierResults(dossier) {
  const pred = dossier.prediction || {};
  const health = dossier.health || {};
  const healthIndex = dossier.health_index || 0.0;
  const recs = dossier.recommendations || [];
  const narrative = dossier.final_narrative || "";
  const mode = dossier.mode || "FIVE_QUESTION";

  // 1. Machine Learning Prediction Badge & Confidence
  const scoreBadge = document.getElementById('res-score-badge');
  const riskTierPill = document.getElementById('res-risk-tier');
  const confText = document.getElementById('res-confidence');
  const modeBadge = document.getElementById('res-mode-badge');

  const predictedClass = pred.predicted_score || "Standard";
  scoreBadge.textContent = predictedClass.toUpperCase();
  scoreBadge.className = `rating-badge-large ${predictedClass}`;

  const tier = pred.risk_tier || "Standard";
  riskTierPill.textContent = tier;
  riskTierPill.className = 'risk-tier-pill';
  if (tier.toLowerCase().includes('low')) riskTierPill.classList.add('low');
  else if (tier.toLowerCase().includes('high')) riskTierPill.classList.add('high');
  else riskTierPill.classList.add('medium');

  confText.textContent = `${((pred.confidence || 0.99) * 100).toFixed(2)}%`;
  modeBadge.textContent = mode === 'FULL' ? 'Full Bureau Pipeline' : '5-Question Engine';

  // Probabilities Spread
  const probs = pred.probabilities || { Good: 0.1, Standard: 0.8, Poor: 0.1 };
  const goodPct = (probs.Good || 0) * 100;
  const stdPct = (probs.Standard || 0) * 100;
  const poorPct = (probs.Poor || 0) * 100;

  document.getElementById('prob-val-good').textContent = `${goodPct.toFixed(2)}%`;
  document.getElementById('prob-bar-good').style.width = `${Math.max(goodPct, 1)}%`;

  document.getElementById('prob-val-standard').textContent = `${stdPct.toFixed(2)}%`;
  document.getElementById('prob-bar-standard').style.width = `${Math.max(stdPct, 1)}%`;

  document.getElementById('prob-val-poor').textContent = `${poorPct.toFixed(2)}%`;
  document.getElementById('prob-bar-poor').style.width = `${Math.max(poorPct, 1)}%`;

  // 2. Credit Health Index Gauge
  const gaugeVal = document.getElementById('health-index-val');
  const gaugeCircle = document.getElementById('gauge-circle');
  const gaugeDesc = document.getElementById('health-status-desc');

  gaugeVal.textContent = healthIndex.toFixed(1);

  // SVG circle calculation (radius 50 -> circumference = 314.15)
  const circumference = 314.15;
  const offset = circumference - (healthIndex / 100) * circumference;
  gaugeCircle.style.strokeDashoffset = offset;

  if (healthIndex >= 80) {
    gaugeCircle.style.stroke = 'var(--color-good)';
    gaugeDesc.textContent = 'Prime Credit Profile (Optimal)';
  } else if (healthIndex >= 55) {
    gaugeCircle.style.stroke = 'var(--color-standard)';
    gaugeDesc.textContent = 'Moderate Underwriting Strain';
  } else {
    gaugeCircle.style.stroke = 'var(--color-poor)';
    gaugeDesc.textContent = 'Elevated Default Hazard';
  }

  // 3. Financial Indicators List
  const indContainer = document.getElementById('indicators-container');
  indContainer.innerHTML = '';

  const indicatorConfigs = [
    { key: 'Debt_to_Income_Ratio', name: 'Debt-to-Income (DTI)', benchmark: 'Benchmark: ≤ 0.35 stable, ≥ 0.55 critical' },
    { key: 'EMI_Burden_Ratio', name: 'EMI-to-Income Burden', benchmark: 'Benchmark: ≤ 0.35 stable, ≥ 0.50 critical' },
    { key: 'Credit_Utilization', name: 'Revolving Credit Utilization', benchmark: 'Benchmark: < 30% stable, > 60% critical', unit: '%' },
    { key: 'Payment_Delay_Days', name: 'Payment Schedule Latency', benchmark: 'Benchmark: ≤ 7d stable, > 18d critical', unit: ' days' },
    { key: 'Hard_Inquiries', name: 'Hard Credit Inquiries (6m)', benchmark: 'Benchmark: ≤ 2 stable, > 6 critical', unit: ' pulls' }
  ];

  indicatorConfigs.forEach(cfg => {
    const item = health[cfg.key] || { value: 0, risk_tier: 'Stable' };
    const row = document.createElement('div');
    row.className = 'indicator-row';
    const displayVal = cfg.unit ? `${item.value}${cfg.unit}` : item.value;

    row.innerHTML = `
      <div class="indicator-meta">
        <span class="indicator-name">${cfg.name}</span>
        <span class="indicator-benchmark">${cfg.benchmark}</span>
      </div>
      <div class="indicator-values">
        <span class="indicator-val">${displayVal}</span>
        <span class="tier-badge ${item.risk_tier}">${item.risk_tier}</span>
      </div>
    `;
    indContainer.appendChild(row);
  });

  // 4. Quantitative Improvement Targets
  const recContainer = document.getElementById('recommendations-container');
  const countBadge = document.getElementById('targets-count');
  recContainer.innerHTML = '';
  countBadge.textContent = `${recs.length} Action${recs.length === 1 ? '' : 's'} Generated`;

  if (recs.length === 0) {
    recContainer.innerHTML = `
      <div class="indicator-row" style="padding: 20px; justify-content: center; color: var(--color-good);">
        All financial metrics are currently within optimal underwriting tolerances. No immediate remediation actions required.
      </div>
    `;
  } else {
    recs.forEach(rec => {
      const card = document.createElement('div');
      card.className = `rec-card ${rec.severity.toLowerCase()}`;
      card.innerHTML = `
        <div class="rec-header">
          <span class="rec-pillar">${rec.pillar}</span>
          <div class="rec-badges">
            <span class="priority-badge">Priority ${rec.priority_rank}</span>
            <span class="severity-tag ${rec.severity}">${rec.severity} Severity</span>
          </div>
        </div>
        <div class="rec-issue">${rec.issue}</div>
        <div class="rec-metrics-row">
          <span class="current">Current: <strong>${rec.current_metric}</strong></span>
          <span class="target">→ Target: <strong>${rec.target_metric}</strong></span>
        </div>
        <div class="rec-action">${rec.action_plan}</div>
        <div class="rec-window">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          Expected Window: ${rec.expected_impact_window}
        </div>
      `;
      recContainer.appendChild(card);
    });
  }

  // 5. Gemini 2.5 Flash / Analytical Narrative Synthesis
  const narrativeBox = document.getElementById('dossier-narrative-text');
  narrativeBox.textContent = narrative;
}

// -------------------------------------------------------------
// Copy Dossier
// -------------------------------------------------------------
function initCopyButton() {
  const btn = document.getElementById('btn-copy-dossier');
  btn.addEventListener('click', () => {
    const text = document.getElementById('dossier-narrative-text').textContent;
    navigator.clipboard.writeText(text).then(() => {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
      setTimeout(() => {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy Dossier`;
      }, 2000);
    });
  });
}

// -------------------------------------------------------------
// Model Analytics & Benchmark Modal
// -------------------------------------------------------------
function initModal() {
  const btnOpen = document.getElementById('btn-toggle-benchmark');
  const btnClose = document.getElementById('btn-close-modal');
  const modal = document.getElementById('modal-benchmarks');

  btnOpen.addEventListener('click', () => modal.classList.remove('hidden'));
  btnClose.addEventListener('click', () => modal.classList.add('hidden'));

  modal.addEventListener('click', e => {
    if (e.target === modal) modal.classList.add('hidden');
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      modal.classList.add('hidden');
    }
  });
}

async function loadBenchmarkMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const metrics = await res.json();

    // 1. Dual-Model comparison table
    const dualTableBody = document.querySelector('#table-dual-model tbody');
    dualTableBody.innerHTML = '';
    (metrics.dual_model_comparison || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row['Model Version']}</strong></td>
        <td>${row['Inputs Required']}</td>
        <td>${row['Macro F1']}</td>
        <td>${row['Accuracy']}</td>
        <td><span class="value-chip">${row['User Friction']}</span></td>
      `;
      dualTableBody.appendChild(tr);
    });

    // 2. Candidate Baseline Models Table
    const candidateTableBody = document.querySelector('#table-candidate-models tbody');
    candidateTableBody.innerHTML = '';
    (metrics.candidate_models || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row['Model']}</strong></td>
        <td>${row['Accuracy']}</td>
        <td>${row['Macro Precision']}</td>
        <td>${row['Macro Recall']}</td>
        <td>${row['Macro F1']}</td>
        <td>${row['ROC-AUC (OvR)']}</td>
        <td>${row['Log Loss']}</td>
      `;
      candidateTableBody.appendChild(tr);
    });

    // 3. Hazard Box
    const hazardBox = document.getElementById('hazard-box');
    const hazard = metrics.underwriting_risk || {};
    hazardBox.innerHTML = `
      <div class="hazard-stat-item">
        <div class="hazard-stat-label">Verified Poor Borrowers in Test Set</div>
        <div class="hazard-stat-val">${hazard.total_poor_tested || 0}</div>
      </div>
      <div class="hazard-stat-item">
        <div class="hazard-stat-label">Catastrophic Approvals (Poor → Good)</div>
        <div class="hazard-stat-val" style="color: var(--color-good);">${hazard.catastrophic_approvals || 0}</div>
      </div>
      <div class="hazard-stat-item">
        <div class="hazard-stat-label">Poor-to-Good Hazard Rate</div>
        <div class="hazard-stat-val" style="color: var(--color-good);">${hazard.hazard_rate_percent || 0}%</div>
      </div>
    `;
  } catch (e) {
    console.error('Failed to load benchmark metrics', e);
  }
}
