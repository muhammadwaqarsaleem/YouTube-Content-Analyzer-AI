/* ── Helpers ──────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const fmt = n => Number(n).toLocaleString();
const pct = v => (v * 100).toFixed(1) + '%';

/* ── Loading step sequencer ──────────────────────────────────────── */
const STEPS = ['metadata','transcript','age','harm','impact','category','violence'];
let stepTimer = null;
function startSteps() {
  let i = 0;
  STEPS.forEach(s => { const el = $('step-'+s); if(el){ el.classList.remove('active','done'); el.querySelector('.step-dot') && (el.querySelector('.step-label').textContent = el.querySelector('.step-label').textContent); } });
  function tick() {
    if (i < STEPS.length) {
      if (i > 0) { const prev = $('step-'+STEPS[i-1]); if(prev) prev.classList.replace('active','done'); }
      const cur = $('step-'+STEPS[i]); if(cur) cur.classList.add('active');
      i++;
      stepTimer = setTimeout(tick, 4000 + Math.random()*2000);
    }
  }
  tick();
}
function stopSteps() { clearTimeout(stepTimer); STEPS.forEach(s => { const el=$('step-'+s); if(el){el.classList.remove('active'); el.classList.add('done');} }); }

/* ── UI State ────────────────────────────────────────────────────── */
function showLoading() {
  $('hero').classList.add('hidden');
  $('resultsLayout').classList.add('hidden');
  $('loadingSection').classList.remove('hidden');
  startSteps();
}
function showResults() {
  stopSteps();
  $('loadingSection').classList.add('hidden');
  $('resultsLayout').classList.remove('hidden');
}
function showHero() {
  $('loadingSection').classList.add('hidden');
  $('resultsLayout').classList.add('hidden');
  $('hero').classList.remove('hidden');
}
function showError(msg) {
  stopSteps();
  $('loadingSection').classList.add('hidden');
  $('hero').classList.remove('hidden');
  $('toastMsg').textContent = msg;
  $('errorToast').classList.remove('hidden');
  setTimeout(() => $('errorToast').classList.add('hidden'), 6000);
}

/* ── Animated bar helper ─────────────────────────────────────────── */
function animateBar(el, pctVal, color) {
  el.style.background = color;
  requestAnimationFrame(() => { el.style.width = Math.min(pctVal * 100, 100) + '%'; });
}

/* ── SVG Gauge ───────────────────────────────────────────────────── */
function buildGauge(score) {
  const r = 72, cx = 90, cy = 90;
  const startAngle = 135, sweep = 270;
  function polar(angle) {
    const rad = (angle - 90) * Math.PI / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  }
  function describeArc(from, to) {
    const [sx, sy] = polar(from), [ex, ey] = polar(to);
    const large = (to - from) > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`;
  }
  const endAngle = startAngle + (score / 100) * sweep;
  const color = score >= 70 ? '#10b981' : score >= 55 ? '#f59e0b' : '#ef4444';
  return `<svg viewBox="0 0 180 180" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="${describeArc(startAngle, startAngle+sweep)}" stroke="rgba(255,255,255,.08)" stroke-width="10" stroke-linecap="round"/>
    <path d="${describeArc(startAngle, endAngle)}" stroke="${color}" stroke-width="10" stroke-linecap="round"
      style="stroke-dasharray:${r*2*Math.PI};stroke-dashoffset:${r*2*Math.PI};transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1);"
      class="gauge-arc"/>
  </svg>`;
}
function animateGauge(container) {
  setTimeout(() => {
    const arc = container.querySelector('.gauge-arc');
    if (!arc) return;
    const r = 72, circ = r * 2 * Math.PI;
    arc.style.strokeDashoffset = '0';
    arc.style.strokeDasharray = circ;
  }, 100);
}

/* ── Color helpers ───────────────────────────────────────────────── */
const AGE_COLORS = { General:'#10b981', Teen:'#f59e0b', Mature:'#ef4444' };
const HARM_COLORS = {
  'Harmless':'#10b981','Clickbait':'#f59e0b','Info Harm':'#fb923c',
  'Physical Harm':'#ef4444','Addiction':'#a855f7','Sexual':'#ec4899','Hate/Harass':'#dc2626'
};
function ageBadgeClass(label) { return label==='General'?'badge-safe':label==='Teen'?'badge-warn':'badge-danger'; }
function harmBadgeClass(label) { return label==='Harmless'?'badge-safe':label==='Clickbait'||label==='Info Harm'?'badge-warn':'badge-danger'; }
function dimColor(val) { return val>=70?'#10b981':val>=55?'#f59e0b':'#ef4444'; }
function scoreColor(s) { return s>=70?'#10b981':s>=55?'#f59e0b':'#ef4444'; }

/* ── Render: Video Summary ───────────────────────────────────────── */
function renderVideo(v) {
  $('videoSummary').innerHTML = `
  <div class="video-summary-grid">
    <div class="video-thumb">
      <img src="${v.thumbnail_url}" alt="Thumbnail" onerror="this.style.display='none'"/>
    </div>
    <div class="video-meta">
      <div class="video-meta-title">${v.title}</div>
      <div class="video-meta-channel">📺 ${v.channel}</div>
      <div class="video-stats">
        <div class="stat-chip"><span class="stat-val">${fmt(v.view_count)}</span><span class="stat-lbl">views</span></div>
        <div class="stat-chip"><span class="stat-val">${fmt(v.like_count)}</span><span class="stat-lbl">likes</span></div>
        <div class="stat-chip"><span class="stat-val">${fmt(v.comment_count)}</span><span class="stat-lbl">comments</span></div>
        <div class="stat-chip"><span class="stat-val">${v.duration}</span><span class="stat-lbl">duration</span></div>
        <div class="stat-chip"><span class="stat-val">${v.published_date}</span><span class="stat-lbl">published</span></div>
      </div>
      <a href="${v.url}" target="_blank" class="video-url-link">🔗 Open on YouTube ↗</a>
    </div>
  </div>`;
}

/* ── Render: Age Classification ──────────────────────────────────── */
function renderAge(data) {
  if (!data) { $('ageContent').innerHTML = '<p style="color:var(--muted)">No data returned.</p>'; return; }
  const probs = data.probabilities || {};
  const sorted = Object.entries(probs).sort((a,b) => b[1]-a[1]);
  const barsHtml = sorted.map(([cls, p]) => {
    const isPred = cls === data.label;
    const color = AGE_COLORS[cls] || '#6366f1';
    return `<div class="prob-row${isPred?' predicted-row':''}">
      <div class="prob-header">
        <span class="prob-label${isPred?' predicted':''}">${cls}${isPred?' ◀':''}</span>
        <span class="prob-pct" style="color:${color}">${pct(p)}</span>
      </div>
      <div class="prob-track"><div class="prob-fill" data-val="${p}" style="background:${color}"></div></div>
    </div>`;
  }).join('');

  const chunksHtml = (data.top3_chunks||[]).map((c,i) => {
    const medals = ['🥇','🥈','🥉'];
    return `<div class="evidence-chunk">
      <div class="evidence-rank">${medals[i]||'#'} Chunk ${c.chunk_idx} — weight ${c.weight}</div>
      <div class="evidence-text">"${c.text}…"</div>
    </div>`;
  }).join('');

  $('ageContent').innerHTML = `
    <div class="verdict-row">
      <span class="verdict-badge ${ageBadgeClass(data.label)}">${data.label}</span>
      <span class="confidence-pill">${pct(data.confidence)} confidence · ${data.num_chunks} chunks analyzed</span>
    </div>
    <div class="prob-bars">${barsHtml}</div>
    ${chunksHtml ? `<div class="accordion" id="ageAccordion">
      <div class="accordion-header" onclick="toggleAccordion('ageAccordion')">
        <span>📝 Top Contributing Transcript Segments (XAI Evidence)</span>
        <span class="accordion-arrow">▾</span>
      </div>
      <div class="accordion-body">${chunksHtml}</div>
    </div>` : ''}`;

  // animate bars
  $('ageContent').querySelectorAll('.prob-fill').forEach(el => {
    animateBar(el, parseFloat(el.dataset.val), el.style.background);
  });
}

/* ── Render: Harm Detection ──────────────────────────────────────── */
function renderHarm(data) {
  if (!data) { $('harmContent').innerHTML = '<p style="color:var(--muted)">No data returned.</p>'; return; }
  const probs = data.probabilities || {};
  const sorted = Object.entries(probs).sort((a,b) => b[1]-a[1]);

  const segsHtml = sorted.map(([cls, p]) => {
    const color = HARM_COLORS[cls]||'#6366f1';
    return `<div class="harm-seg" data-val="${p}" style="background:${color};width:0"></div>`;
  }).join('');

  const legendHtml = sorted.map(([cls, p]) => {
    const color = HARM_COLORS[cls]||'#6366f1';
    const isPred = cls === data.label;
    return `<div class="harm-legend-item">
      <div class="harm-dot" style="background:${color}"></div>
      <span style="${isPred?'color:var(--text);font-weight:600':''}">${cls}</span>
      <span class="harm-pct" style="color:${color}">${pct(p)}</span>
    </div>`;
  }).join('');

  const barsHtml = sorted.map(([cls, p]) => {
    const color = HARM_COLORS[cls]||'#6366f1';
    const isPred = cls === data.label;
    return `<div class="prob-row${isPred?' predicted-row':''}">
      <div class="prob-header">
        <span class="prob-label${isPred?' predicted':''}">${cls}${isPred?' ◀ FLAGGED':''}</span>
        <span class="prob-pct" style="color:${color}">${pct(p)}</span>
      </div>
      <div class="prob-track"><div class="prob-fill" data-val="${p}" style="background:${color}"></div></div>
    </div>`;
  }).join('');

  $('harmContent').innerHTML = `
    <div class="verdict-row">
      <span class="verdict-badge ${harmBadgeClass(data.label)}">${data.label}</span>
      <span class="confidence-pill">${pct(data.confidence)} confidence</span>
    </div>
    <div class="harm-stacked" id="harmStacked">${segsHtml}</div>
    <div class="harm-legend">${legendHtml}</div>
    <div class="dim-section-title" style="margin-top:1.5rem">Risk Breakdown</div>
    <div class="prob-bars">${barsHtml}</div>`;

  // animate stacked bar
  const total = sorted.reduce((s,[,p])=>s+p, 0)||1;
  $('harmContent').querySelectorAll('.harm-seg').forEach(el => {
    const val = parseFloat(el.dataset.val)/total*100;
    requestAnimationFrame(() => { el.style.width = val + '%'; });
  });
  $('harmContent').querySelectorAll('.prob-fill').forEach(el => {
    animateBar(el, parseFloat(el.dataset.val), el.style.background);
  });
}

/* ── Render: Impact ──────────────────────────────────────────────── */
function renderImpact(data) {
  if (!data) { $('impactContent').innerHTML = '<p style="color:var(--muted)">No data returned.</p>'; return; }
  const s = data.score, level = data.level;
  const color = scoreColor(s);
  const gaugeSvg = buildGauge(s);

  const levelMap = {MINIMAL:'0–39',LOW:'40–54',MODERATE:'55–69',HIGH:'70–79',VIRAL:'80–100'};
  const levelRows = Object.entries(levelMap).map(([l,r]) =>
    `<div class="impact-level-row${l===level?' active':''}"><span class="ilr-label">${l}</span><span class="ilr-range">${r}</span></div>`).join('');

  const dimWeights = {quality:30,engagement:25,sentiment:20,reach:15,virality:10};
  const dimSubLabels = {
    quality:'Transcript depth · Title/Metadata · Structural',
    engagement:'Rate quality · Signal · Momentum',
    sentiment:'Valence · Intensity · Polarization',
    reach:'Scale · Velocity · Relative performance',
    virality:'Historical lift · Current velocity · Shareability'
  };
  const dimsHtml = Object.entries(data.dimension_scores||{}).map(([dim, val]) => {
    const c = dimColor(val);
    return `<div class="dim-row">
      <div class="dim-header">
        <span class="dim-name">${dim.charAt(0).toUpperCase()+dim.slice(1)}</span>
        <div class="dim-right"><span class="dim-weight">wt ${dimWeights[dim]||0}%</span><span class="dim-val" style="color:${c}">${val.toFixed(1)}</span></div>
      </div>
      <div class="dim-track"><div class="dim-fill" data-val="${val/100}" style="background:${c}"></div></div>
      <div class="dim-sublabel">${dimSubLabels[dim]||''}</div>
    </div>`;
  }).join('');

  const factorsHtml = (data.key_factors||[]).map(f =>
    `<div class="factor-item"><span class="factor-arrow">▲</span><span>${f}</span></div>`).join('');

  const ta = data.transcript_analysis||{};
  const qStars = Math.max(1,Math.round((ta.quality_score||0)*5));
  const tqHtml = `
    <div class="tq-grid">
      <div class="tq-chip"><div class="tq-chip-label">Word Count</div><div class="tq-chip-val">${fmt(ta.word_count||0)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Vocab Richness</div><div class="tq-chip-val">${(ta.vocab_richness||0).toFixed(3)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Avg Sentence Len</div><div class="tq-chip-val">${(ta.avg_sentence_len||0).toFixed(1)} words</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Semantic Depth</div><div class="tq-chip-val">${(ta.semantic_similarity||0).toFixed(3)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Filler Density</div><div class="tq-chip-val">${(ta.filler_penalty||0).toFixed(3)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Linguistic Score</div><div class="tq-chip-val">${(ta.linguistic_score||0).toFixed(3)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Quality Score</div><div class="tq-chip-val tq-stars">${'★'.repeat(qStars)+'☆'.repeat(5-qStars)} ${(ta.quality_score||0).toFixed(3)}</div></div>
      <div class="tq-chip"><div class="tq-chip-label">Transcript</div><div class="tq-chip-val">${ta.has_content?'✅ Present':'❌ None'}</div></div>
    </div>`;

  const reasonParts = (data.reasoning||'').split('|').map(p=>p.trim()).filter(Boolean);
  const reasonHtml = reasonParts.map(p=>`<div class="factor-item"><span class="factor-arrow">•</span><span>${p}</span></div>`).join('');

  $('impactContent').innerHTML = `
    <div class="gauge-wrap">
      <div class="gauge-svg-container" id="gaugeContainer">
        ${gaugeSvg}
        <div class="gauge-center-text">
          <div class="gauge-score" style="color:${color}">${s.toFixed(1)}</div>
          <div class="gauge-level">${level}</div>
        </div>
      </div>
      <div class="gauge-meta">
        <div class="gauge-meta-title">Overall Impact Score</div>
        ${data.excellence_bonus>0?`<div class="gauge-meta-bonus">✨ +${data.excellence_bonus.toFixed(1)} excellence bonus</div>`:''}
        <div class="impact-levels">${levelRows}</div>
      </div>
    </div>
    <div class="dim-section-title">📈 Dimension Scores</div>
    <div class="dim-bars">${dimsHtml}</div>
    ${factorsHtml?`<div class="dim-section-title">🔑 Key Factors</div><div class="factors-list">${factorsHtml}</div>`:''}
    <div class="dim-section-title">📝 Transcript & Content Quality</div>${tqHtml}
    ${reasonHtml?`<div class="dim-section-title">💡 Analysis Reasoning</div><div class="factors-list">${reasonHtml}</div>`:''}`;

  animateGauge($('gaugeContainer'));
  $('impactContent').querySelectorAll('.dim-fill').forEach(el => animateBar(el, parseFloat(el.dataset.val), el.style.background));
}

/* ── Render: Category ────────────────────────────────────────────── */
function renderCategory(data) {
  if (!data) { $('categoryContent').innerHTML = '<p style="color:var(--muted)">No data returned.</p>'; return; }
  const cats = (data.all_categories||[]).slice(0,7);
  const barsHtml = cats.map(c => {
    const isPred = c.category === data.primary;
    const color = isPred ? '#06b6d4' : '#6366f1';
    return `<div class="prob-row${isPred?' predicted-row':''}">
      <div class="prob-header">
        <span class="prob-label${isPred?' predicted':''}">${c.category}${isPred?' ◀':''}</span>
        <span class="prob-pct" style="color:${color}">${pct(c.probability)}</span>
      </div>
      <div class="prob-track"><div class="prob-fill" data-val="${c.probability}" style="background:${color}"></div></div>
    </div>`;
  }).join('');

  $('categoryContent').innerHTML = `
    <div class="verdict-row">
      <span class="verdict-badge badge-info">${data.primary}</span>
      <span class="confidence-pill">${pct(data.primary_probability)} confidence</span>
    </div>
    <div class="prob-bars">${barsHtml}</div>`;

  $('categoryContent').querySelectorAll('.prob-fill').forEach(el => animateBar(el, parseFloat(el.dataset.val), el.style.background));
}

/* ── Render: Violence ────────────────────────────────────────────── */
function renderViolence(data) {
  if (!data) { $('violenceContent').innerHTML = '<p style="color:var(--muted)">No data returned.</p>'; return; }
  const isV = data.is_violent;
  const tierLabels = {1:'Tier 1 — yt-dlp android/tv_embedded',2:'Tier 2 — yt-dlp ios/web_embedded',3:'Tier 3 — YouTube storyboard scraping',4:'Tier 4 — Static thumbnail fallback'};
  const tsChips = (data.violent_timestamps||[]).map(ts => {
    const m = Math.floor(ts/60), s = Math.floor(ts%60);
    return `<span class="ts-chip">${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}</span>`;
  }).join('');

  $('violenceContent').innerHTML = `
    <div class="violence-status ${isV?'danger':'safe'}">
      <div class="violence-status-icon">${isV?'🟥':'🟩'}</div>
      <div>
        <div class="violence-status-label">${isV?'Violent Content Detected':'No Significant Violence'}</div>
        <div class="violence-status-sub">Severity: ${data.severity} · ${tierLabels[data.tier_used]||'Unknown tier'}</div>
      </div>
    </div>
    <div class="violence-stats">
      <div class="v-stat"><div class="v-stat-label">Violence %</div><div class="v-stat-val">${data.violence_percentage.toFixed(2)}%</div></div>
      <div class="v-stat"><div class="v-stat-label">Violent Frames</div><div class="v-stat-val">${fmt(data.violent_frame_count)} / ${fmt(data.total_frames)}</div></div>
      <div class="v-stat"><div class="v-stat-label">Peak Confidence</div><div class="v-stat-val">${pct(data.peak_confidence)}</div></div>
      <div class="v-stat"><div class="v-stat-label">Severity Level</div><div class="v-stat-val">${data.severity}</div></div>
    </div>
    ${tsChips?`<div class="dim-section-title">🕒 Detected Violence Timestamps</div><div class="timestamps-list">${tsChips}</div>`:''}
    ${data.recommendation?`<div class="recommendation-box">📋 ${data.recommendation}</div>`:''}
    ${data.tier_used===4?'<div class="recommendation-box" style="border-color:rgba(245,158,11,.3);color:#fbbf24">⚠️ Analysis ran on static thumbnail frames only — video streams unavailable.</div>':''}`;
}

/* ── Sidebar scroll spy ──────────────────────────────────────────── */
function initScrollSpy() {
  const sections = ['sec-video','sec-age','sec-harm','sec-impact','sec-category','sec-violence'];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
        const link = document.querySelector(`.sidebar-link[data-section="${e.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { threshold: 0.3 });
  sections.forEach(id => { const el = $(id); if(el) observer.observe(el); });
}

/* ── Accordion ───────────────────────────────────────────────────── */
function toggleAccordion(id) {
  const el = $(id);
  if (el) el.classList.toggle('open');
}
window.toggleAccordion = toggleAccordion;

/* ── Main fetch & render ─────────────────────────────────────────── */
$('analyzeForm').addEventListener('submit', async e => {
  e.preventDefault();
  const url = $('urlInput').value.trim();
  if (!url) return;
  $('analyzeBtn').disabled = true;
  $('loadingVideoTitle').textContent = 'Analyzing: ' + url;
  showLoading();

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Analysis failed.'); $('analyzeBtn').disabled=false; return; }

    renderVideo(data.video || {});
    renderAge(data.age_classification);
    renderHarm(data.harm_detection);
    renderImpact(data.impact);
    renderCategory(data.category);
    renderViolence(data.violence);
    showResults();
    initScrollSpy();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    $('analyzeBtn').disabled = false;
  }
});

$('newAnalysisBtn').addEventListener('click', () => {
  $('urlInput').value = '';
  showHero();
});
$('toastClose').addEventListener('click', () => $('errorToast').classList.add('hidden'));
