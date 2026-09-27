import { $, bars, esc, fail, kpis, legend, load, pct, pts, seg, table, usd, xy } from './kit.js';

const SHORT = { 'gemini-1.5-flash': 'Gemini 1.5 Flash', 'llama-3.1-8b': 'Llama 3.1 8B', 'gpt-4o-mini': 'GPT-4o mini', 'deepseek-v3': 'DeepSeek V3', 'llama-3.1-70b': 'Llama 3.1 70B', 'gpt-4o': 'GPT-4o' };
const name = (m) => SHORT[m] || m;

try {
  const data = await load();
  let reg, vol = 1e6;
  const regs = Object.keys(data.registries);

  function draw() {
    const g = data.registries[reg], f = g.single[g.frontier], r = g.router;
    kpis($('#kpis'), [
      { label: 'Cost saved vs GPT-4o', value: pct(r.saving), note: `${usd(r.per_1000)} instead of ${usd(f.per_1000)} per 1,000 requests` },
      { label: 'Accuracy vs GPT-4o', value: `${pts(r.accuracy - f.accuracy)} pts`, note: `95% interval ${pts(r.vs_frontier_ci[0])} to ${pts(r.vs_frontier_ci[1])}` },
      { label: 'Held-out questions', value: g.test_questions.toLocaleString('en-US'), note: `${Object.keys(data.tasks).length} HELM tasks, ${data.questions.toLocaleString('en-US')} in all` },
      { label: 'Same answer as GPT-4o', value: pct(r.agrees_with_frontier), note: 'right or wrong on the same questions' },
    ]);
    const singles = Object.entries(g.single).map(([m, s]) => ({ x: s.per_1000, y: s.accuracy, label: name(m), title: `everything to ${m}` }));
    const series = [
      { name: 'one model for everything', color: 'var(--c6)', line: false, dots: true, points: singles },
      { name: 'router at each confidence', color: 'var(--accent)', points: r.curve.map((c) => ({ x: c.per_1000, y: c.accuracy })), dots: false },
      { name: 'learned router (chosen)', color: 'var(--accent)', line: false, dots: true, points: [{ x: r.per_1000, y: r.accuracy, label: 'Router', keep: true, r: 7, title: `learned router, confidence ${r.confidence}` }] },
      { name: `cascade (${g.cascade.pair.map(name).join(' + ')})`, color: 'var(--c2)', line: false, dots: true, points: [{ x: g.cascade.per_1000, y: g.cascade.accuracy, label: 'Cascade', title: `cascade ${g.cascade.pair.join(' + ')}, escalates ${pct(g.cascade.escalated)} to ${g.frontier}` }] },
      { name: 'fixed task-to-model map', color: 'var(--c3)', line: false, dots: true, points: [{ x: g.task_map.per_1000, y: g.task_map.accuracy, label: 'Task map', title: 'fixed task-to-model map' }] },
      { name: 'oracle (hindsight ceiling)', color: 'var(--c4)', line: false, dots: true, points: [{ x: g.oracle.per_1000, y: g.oracle.accuracy, label: 'Oracle', keep: true, title: 'oracle: cheapest model that was right' }] },
    ];
    xy($('#scatter'), {
      label: 'Accuracy against cost per 1,000 requests', height: 340, series,
      x: { log: true, label: 'dollars per 1,000 requests (log scale)', fmt: (v) => `$${v}`, min: 0.08, max: 5 },
      y: { fmt: (v) => `${Math.round(v * 100)}%`, min: 0.45, max: 0.95 },
      onPick: (p, s) => {
        $('#pick').innerHTML = `<div>${esc(p.title || s.name)}<b>${pct(p.y)} right</b></div><div>Cost per 1,000 requests<b>${usd(p.x)}</b></div>` +
          `<div>Against GPT-4o<b>${pts(p.y - f.accuracy)} pts, ${pct(1 - p.x / f.per_1000, 0)} cheaper</b></div>`;
      },
    });
    legend($('#scatterKey'), series.filter((s) => s.name !== 'learned router (chosen)'));
    $('#pick').innerHTML = '';

    const conf = $('#conf');
    conf.max = r.curve.length - 1;
    const chosen = r.curve.findIndex((c) => Math.abs(c.confidence - r.confidence) < 1e-9);
    if (!conf.dataset.touched) conf.value = chosen;
    const showConf = () => {
      const c = r.curve[+conf.value];
      $('#confVal').textContent = `${c.confidence.toFixed(2)}${+conf.value === chosen ? ' (chosen)' : ''}`;
      const month = (x) => `$${Math.round((x * vol) / 1000).toLocaleString('en-US')}`;
      $('#confOut').innerHTML = `<div>Accuracy<b>${pct(c.accuracy)}</b><span class="muted small">GPT-4o alone: ${pct(f.accuracy)}</span></div>` +
        `<div>Per 1,000 requests<b>${usd(c.per_1000)}</b><span class="muted small">GPT-4o alone: ${usd(f.per_1000)}</span></div>` +
        `<div>A month at ${(vol / 1e6).toLocaleString('en-US')}M requests<b>${month(c.per_1000)}</b><span class="muted small">GPT-4o alone: ${month(f.per_1000)}</span></div>`;
    };
    conf.oninput = () => { conf.dataset.touched = '1'; showConf(); };
    showConf();
    seg($('#vol'), [['100000', '100k'], ['1000000', '1M'], ['10000000', '10M']], String(vol), (v) => { vol = +v; showConf(); });

    $('#shareSub').textContent = `Share of prompts per model at the chosen confidence (${r.confidence}).`;
    bars($('#share'), Object.entries(r.share).map(([m, v]) => ({ label: name(m), value: v, color: m === g.frontier ? 'var(--c6)' : 'var(--accent)' })), { max: 1, fmt: (v) => pct(v) });
    table($('#tasks'), [
      { key: 'task', label: 'Task' },
      { key: 'router', label: 'Router', num: true, fmt: (v) => pct(v) },
      { key: 'frontier', label: 'GPT-4o', num: true, fmt: (v) => pct(v) },
      { key: 'per_1000', label: 'Router $/1k', num: true, fmt: (v) => usd(v) },
      { key: 'frontier_per_1000', label: 'GPT-4o $/1k', num: true, fmt: (v) => usd(v) },
    ], Object.entries(r.by_task).map(([task, t]) => ({ task, ...t })));
  }

  seg($('#reg'), regs.map((k) => [k, k === 'any provider' ? 'Any provider' : 'US providers only']), regs.includes('any provider') ? 'any provider' : regs[0], (v) => { reg = v; $('#conf').dataset.touched = ''; draw(); });
} catch (err) {
  fail(err);
}
