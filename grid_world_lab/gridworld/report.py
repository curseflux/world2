"""Write a dependency-free, interactive experiment report.

The report deliberately stores the raw events as well as full-run metrics. Its
map controls select events; they do not silently change the saved experiment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _finite(value: Any) -> Any:
    """Keep browser JSON valid when an undefined aggregate was represented by NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def render_report(payload: dict[str, Any], path: str | Path) -> None:
    """Embed ``payload`` into a portable HTML report at ``path``.

    No web server, JavaScript framework, network connection, or original
    experiment repository is required to open the result.
    """
    data = json.dumps(_finite(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # A JSON script element is still parsed by HTML: a literal </script> would
    # terminate it even though its MIME type is application/json.
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        data = data.replace(character, escaped)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__EXPERIMENT_JSON__", data), encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid world reconstruction · Experiment report</title>
<style>
:root{color-scheme:light;--ink:#192c42;--muted:#617489;--line:#dce5ee;--paper:#fff;--background:#f3f6fa;--blue:#2563a6;--good:#176f93;--mismatch:#bd7110;--illegal:#ca4264;--pale:#c8d3e0;--shadow:0 7px 24px #10274307}
*{box-sizing:border-box}body{margin:0;background:var(--background);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button,select,input[type=text],input[type=number]{border:1px solid #c6d3df;border-radius:7px;background:#fff;color:var(--ink);padding:8px 10px}button{cursor:pointer;font-weight:600}button:hover{border-color:var(--blue);background:#f1f7ff}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #aacdf7;outline-offset:2px}input[type=range]{accent-color:var(--blue);width:100%;cursor:pointer}input[type=checkbox]{accent-color:var(--blue)}select{max-width:100%}h1{font-size:clamp(25px,4vw,37px);line-height:1.18;letter-spacing:-1px;font-weight:700;margin:8px 0 12px}h2{font-size:18px;letter-spacing:-.2px;margin:0}h3{font-size:15px;margin:0}p{margin:7px 0}.muted{color:var(--muted)}.small{font-size:12px}.mono,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.wrap{max-width:1540px;margin:auto;padding:26px 30px 40px}.hero{display:flex;gap:24px;justify-content:space-between;align-items:center;padding:6px 0 20px}.hero-copy{max-width:910px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:2px;color:var(--blue);text-transform:uppercase}.badge{display:inline-flex;align-items:center;gap:7px;border:1px solid #bdd4e9;background:#edf6ff;color:#204f7d;border-radius:99px;padding:5px 11px;white-space:nowrap;font-size:12px}.badge:before{content:"";width:6px;height:6px;border-radius:50%;background:#377fb7}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.panel{padding:18px 20px;margin-bottom:18px}.control-grid{display:grid;grid-template-columns:190px minmax(250px,1fr) minmax(260px,1fr);gap:24px;align-items:start}.field-label{display:block;font-weight:650;font-size:12px;margin-bottom:7px}.range-head{display:flex;justify-content:space-between;gap:15px}.subset-row{display:flex;gap:8px}.subset-row input{width:100%;min-width:0}.filter-row{display:flex;align-items:center;flex-wrap:wrap;gap:12px 23px;padding-top:14px;margin-top:13px;border-top:1px solid var(--line)}.filter-row label{display:flex;gap:6px;align-items:center;font-size:12px}.legend-mark{width:24px;display:inline-block;border-top:2px solid var(--good)}.legend-mark.orange{border-color:var(--mismatch);border-top-style:dashed}.legend-mark.red{border-color:var(--illegal);border-top-style:dashed}.legend-mark.gray{border-color:var(--pale)}.confidence-control input{width:65px;padding:4px 6px}.status{min-height:19px;margin-top:7px;color:var(--muted);font-size:12px}.status.error{color:#b1264c}.metrics-row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-bottom:18px}.metric{padding:13px 15px}.metric .value{font-size:26px;line-height:1.2;font-weight:700;letter-spacing:-.6px;margin-top:5px}.metric .label{font-size:11px;color:var(--muted);font-weight:650}.metric .help{font-size:11px;color:var(--muted);margin-top:5px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}.map-card{min-width:0;overflow:hidden}.map-heading{padding:16px 18px 11px;display:flex;justify-content:space-between;align-items:flex-start;gap:10px}.map-heading p{font-size:12px;color:var(--muted)}.map-stage{padding:0 12px;min-height:250px}.map-stage svg{display:block;width:100%;max-height:630px;aspect-ratio:1.08;touch-action:manipulation}.map-footer{padding:10px 18px 14px;color:var(--muted);font-size:12px;border-top:1px solid #f0f3f7;min-height:44px}.map-stage svg .interactive{cursor:help}.map-stage svg .interactive:hover{filter:drop-shadow(0 0 2px #20456466)}.map-stage svg .graph-node:hover{stroke:var(--ink);stroke-width:2}.section-head{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:12px}.section-head .actions{display:flex;gap:9px;align-items:center}.scroll{overflow:auto;max-height:390px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}th{background:#f4f7fb;color:#56697f;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;position:sticky;top:0;z-index:1}td,th{padding:9px 10px;border-bottom:1px solid #e8edf3;white-space:nowrap}td.wrappable{white-space:normal;max-width:230px}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f6f9fd}.state{font-size:10px;border-radius:4px;padding:2px 5px;font-weight:650}.state.good{background:#e5f3f5;color:#226175}.state.orange{background:#fff1dd;color:#925800}.state.red{background:#fdebf0;color:#ab3454}.state.missed{background:#edf0f5;color:#67748a}.lower-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.frequency-tools{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}.frequency-tools button.active{background:#e7f1ff;border-color:#91b4de}.frequency-tools select{margin-left:auto;font-size:12px}.note{padding:12px 14px;background:#f3f7fc;border-left:3px solid #7fa9d4;border-radius:5px;color:#465c76;font-size:12px;margin:12px 0}.note.warning{background:#fff8e9;border-color:#d8ad55;color:#786035}.mini-chart svg{display:block;width:100%;height:200px}.chart-legend{display:flex;gap:16px;font-size:11px;color:var(--muted);flex-wrap:wrap}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.details-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}details{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}summary{padding:12px 14px;font-size:12px;font-weight:650;cursor:pointer}details>div{padding:0 14px 14px}pre{font-size:11px;line-height:1.55;white-space:pre-wrap;word-break:break-word;background:#f7f9fc;padding:12px;border-radius:5px;max-height:460px;overflow:auto;margin:0}.details-grid>details:last-child:nth-child(odd){grid-column:1/-1}.empty{color:var(--muted);text-align:center;padding:24px 12px}.progress-label{font-size:11px;font-weight:650;color:var(--blue)}.footer{font-size:11px;color:var(--muted);padding-top:12px;text-align:center}.full-metric-columns{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.full-metric-columns h3{margin:8px 0 10px}.full-metric-columns th{position:static}.full-metric-columns td:first-child{white-space:normal}.nowrap{white-space:nowrap}.step-sample{max-width:430px}.sample-meta{font-size:12px;color:var(--muted);margin-bottom:12px;overflow-wrap:anywhere}.tables-note{font-size:11px;color:var(--muted);margin-top:9px}.hover-readout{font-size:12px;min-height:35px;color:var(--muted);border:1px solid var(--line);background:#f9fbfe;padding:8px 11px;border-radius:7px;margin-top:12px}.good-text{color:var(--good)}.orange-text{color:var(--mismatch)}.red-text{color:var(--illegal)}
@media(max-width:1100px){.metrics-row{grid-template-columns:repeat(3,1fr)}.control-grid{grid-template-columns:170px 1fr}.subset-field{grid-column:1/-1}.lower-grid{grid-template-columns:1fr}.full-metric-columns{grid-template-columns:1fr}.hero>.badge{display:none}}
@media(max-width:720px){.wrap{padding:18px 13px 26px}.hero{padding-bottom:12px}.control-grid,.maps,.details-grid{grid-template-columns:1fr}.subset-field{grid-column:auto}.maps{gap:12px}.metrics-row{grid-template-columns:repeat(2,1fr);gap:8px}.metric{padding:11px}.panel{padding:15px 13px}.section-head{align-items:flex-start;flex-direction:column}.section-head .actions{width:100%;flex-wrap:wrap}.step-sample{max-width:100%}.filter-row{gap:10px 14px}.map-stage svg{max-height:520px}.details-grid>details:last-child:nth-child(odd){grid-column:auto}h1{letter-spacing:-.5px}}
@media print{.control-panel,.footer button{display:none}.wrap{max-width:none;padding:0}.card{box-shadow:none;break-inside:avoid}.scroll{max-height:none;overflow:visible}.maps{grid-template-columns:1fr 1fr}.map-stage svg{max-height:350px}details{break-inside:avoid}}
</style>
</head>
<body>
<main class="wrap">
  <header class="hero">
    <div class="hero-copy"><div class="eyebrow">Generative world models / Grid experiment</div><h1>What map does the model reveal?</h1><p class="muted">Follow generated directions, then let the location probe set the next node. Inspect the geometry, the mistakes, and the training evidence behind each edge.</p></div>
    <span class="badge">Portable offline report</span>
  </header>
  <section class="card panel control-panel" aria-label="Map controls">
    <div class="control-grid">
      <div><label class="field-label" for="cohort">Test cohort</label><select id="cohort"><option value="all">Both cohorts</option><option value="seen">Seen endpoint pairs</option><option value="unseen">Unseen endpoint pairs</option></select><div id="cohort-status" class="status"></div></div>
      <div><div class="range-head"><label class="field-label" for="sample-count">Cumulative samples</label><output id="range-value" class="progress-label">0 / 0</output></div><input id="sample-count" type="range" min="0" max="0" value="0" step="1"><div class="status">The first <span id="count-description">0</span> samples in this cohort, in saved order.</div></div>
      <div class="subset-field"><label class="field-label" for="sample-subset">Or select specific samples</label><div class="subset-row"><input id="sample-subset" type="text" placeholder="1, 3–7, or exact sample IDs" autocomplete="off"><button id="clear-subset" title="Return to cumulative slider">Clear</button></div><div id="subset-status" class="status">Numbers are 1-based positions within the selected cohort.</div></div>
    </div>
    <div class="filter-row">
      <label><input id="show-correct" type="checkbox" checked><span class="legend-mark"></span>Legal + probe agrees</label>
      <label><input id="show-mismatch" type="checkbox" checked><span class="legend-mark orange"></span>Legal + probe disagrees</label>
      <label><input id="show-illegal" type="checkbox" checked><span class="legend-mark red"></span>Illegal direction</label>
      <label><input id="show-missing" type="checkbox" checked>Missing true edges</label>
      <label><input id="show-labels" type="checkbox" checked>Node IDs</label>
      <label><input id="show-frequency" type="checkbox" checked>Weight by frequency</label>
      <label class="confidence-control">Show events with probe confidence ≥ <input id="min-confidence" type="number" min="0" max="1" step="0.05" value="0"></label>
    </div>
    <p class="small muted">Edge-type and confidence filters affect drawing only. Metrics and frequency tables use every event in the selected samples.</p>
  </section>
  <div id="probe-diagnostic" class="note"></div>
  <div id="selection-metrics" class="metrics-row" aria-live="polite"></div>
  <div class="maps">
    <section class="card map-card"><div class="map-heading"><div><h2>True map</h2><p id="true-map-subtitle"></p></div><span class="badge" id="grid-size"></span></div><div class="map-stage" id="true-map"></div><div class="map-footer">Only nodes visited and undirected edges traversed during training exist. Hover for training counts.</div></section>
    <section class="card map-card"><div class="map-heading"><div><h2>Reconstructed map</h2><p id="recon-map-subtitle"></p></div><span class="badge" id="selection-badge"></span></div><div class="map-stage" id="recon-map"></div><div class="map-footer" id="recon-footer"></div></section>
  </div>
  <div id="hover-readout" class="hover-readout">Hover over a node or edge to inspect it. Dotted edges remain in the map; generation continues from the post-direction probe prediction.</div>
  <div class="lower-grid" style="margin-top:18px">
    <section class="card panel"><div class="section-head"><div><h2>Coverage as samples accumulate</h2><p class="small muted">Saved cohort order; curves always use the complete cohort.</p></div></div><div id="coverage-chart" class="mini-chart"></div><div class="chart-legend"><span><span class="dot" style="background:var(--good)"></span>True-edge recall</span><span><span class="dot" style="background:#7c68b0"></span>Node coverage</span><span><span class="dot" style="background:var(--illegal)"></span>Fake edges / true edges</span></div><p class="tables-note">The last ratio measures added topology relative to the size of the real graph. It can exceed 100%.</p></section>
    <section class="card panel"><div class="section-head"><h2>How to read the errors</h2></div><p><span class="good-text"><b>Solid:</b></span> the direction is legal at the current inferred node and the post-direction probe predicts its true neighbor.</p><p><span class="orange-text"><b>Orange dotted:</b></span> the direction is legal, but the probe predicts a different resulting node.</p><p><span class="red-text"><b>Red dotted:</b></span> no true edge exists for that direction at the current inferred node.</p><div class="note">An <b>action/location error</b> is any dotted event. A <b>fake topology edge</b> is a source–target pair absent from the undirected true graph, including self-loops. A dotted event can connect two nodes that really are adjacent, so these are different measurements.</div><div class="note warning">Probe confidence is a classifier score. Validation on legal held-out walks does not establish location accuracy or calibration after an illegal prefix. The report records the inferred state, not a verified hidden intention.</div></section>
  </div>
  <section class="card panel"><div class="section-head"><div><h2>Follow one generated sample</h2><p class="small muted">Every direction is consumed before probing. The predicted node becomes the source for the next step.</p></div><div class="actions"><select id="detail-sample" class="step-sample" aria-label="Sample details"></select><button id="export-events">Export selected events CSV</button></div></div><div id="sample-meta" class="sample-meta"></div><div class="scroll"><table><thead><tr><th>Step</th><th>Source</th><th>Direction</th><th>True neighbor</th><th>Probe node</th><th>Event</th><th>Probe confidence</th><th>P(direction)</th><th>Probe alternatives</th></tr></thead><tbody id="step-body"></tbody></table></div></section>
  <section class="card panel"><div class="section-head"><div><h2>What was remembered, and what is missing?</h2><p class="small muted">Training evidence compared with the currently selected reconstructions and their legal reference walks.</p></div><button id="export-frequency">Export frequency table CSV</button></div><div class="frequency-tools"><button id="frequency-edges" class="active">Edges</button><button id="frequency-nodes">Nodes</button><label class="small"><input id="only-missing" type="checkbox">Only missing</label><select id="frequency-sort" aria-label="Frequency table order"><option value="missing">Frequent but missing first</option><option value="rare">Rarest in training first</option><option value="frequent">Most frequent in training first</option><option value="generated">Most used by model first</option></select></div><div class="scroll"><table><thead id="frequency-head"></thead><tbody id="frequency-body"></tbody></table></div><p class="tables-note">Reconstructed counts include all selected events connecting that true edge (or visiting that node). Solid counts are stricter. A missing edge is not automatically a wrong prediction: endpoint exposure and route choice affect its opportunity to appear. Reference counts help make that exposure visible.</p></section>
  <section class="card panel"><div class="section-head"><div><h2>Full experiment results</h2><p class="small muted">Saved metrics for all generated samples. These tables do not change with the slider or display filters.</p></div></div><div id="full-metrics" class="full-metric-columns"></div></section>
  <section class="card panel"><div class="section-head"><div><h2>Experiment details</h2><p class="small muted">Configuration, training history, probe validation, and metric definitions are embedded in this file.</p></div><button id="export-json">Download experiment JSON</button></div><div id="experiment-details" class="details-grid"></div></section>
  <footer class="footer">Standalone grid-world experiment · No external resources are loaded · Node IDs use row-major order, starting at 1</footer>
</main>
<script id="experiment-data" type="application/json">__EXPERIMENT_JSON__</script>
<script>
"use strict";
(() => {
const data = JSON.parse(document.getElementById("experiment-data").textContent);
const graph = data.dataset?.graph || {rows:1,cols:1,nodes:[],edges:[],node_counts:{},edge_counts:{}};
const samples = Array.isArray(data.samples) ? data.samples : [];
const rows = Number(graph.rows) || 1, cols = Number(graph.cols) || 1;
const nodes = (graph.nodes || []).map(Number);
const edges = (graph.edges || []).map(e => [Number(e[0]),Number(e[1])]);
const key = (a,b) => Number(a)<=Number(b) ? `${Number(a)}-${Number(b)}` : `${Number(b)}-${Number(a)}`;
const trueEdges = new Set(edges.map(e=>key(...e))), trueNodes = new Set(nodes);
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = (x,digits=2) => x===null||x===undefined||!Number.isFinite(Number(x)) ? "—" : Number(x).toLocaleString(undefined,{maximumFractionDigits:digits});
const pct = x => x===null||x===undefined||!Number.isFinite(Number(x)) ? "—" : `${(100*Number(x)).toFixed(1)}%`;
const mean = xs => xs.length ? xs.reduce((s,v)=>s+Number(v||0),0)/xs.length : null;
const ratio = (a,b) => b ? a/b : null;
const eventsOf = sample => Array.isArray(sample.events) ? sample.events : [];
const sampleId = (s,i) => String(s.id ?? `sample-${i+1}`);
const idOf = new Map(samples.map((s,i)=>[s,sampleId(s,i)]));
const kindNames = {legal_correct:"Legal + agrees",legal_mismatch:"Legal + disagrees",illegal:"Illegal direction"};
const kindColors = {legal_correct:"#176f93",legal_mismatch:"#bd7110",illegal:"#ca4264"};
let pool = samples, selected = [], currentAnalysis = null, frequencyMode = "edges", frequencyRows = [], coveragePoints = [];

function counted(map,k,n=1){map.set(k,(map.get(k)||0)+n);}
function analyze(selectedSamples){
 const reconstructed = new Set(), seenNodes = new Set(), fake = new Set(), recovered = new Set(), solid = new Set();
 const edgeCounts = new Map(), solidCounts = new Map(), nodeCounts = new Map(), targetCounts = new Map(), originCounts = new Map(), referenceEdges = new Map(), referenceNodes = new Map();
 let steps=0,errors=0,illegal=0,mismatch=0,fakeEvents=0;
 for(const sample of selectedSamples){
  if(sample.origin!==null && sample.origin!==undefined){seenNodes.add(Number(sample.origin));counted(nodeCounts,Number(sample.origin));counted(originCounts,Number(sample.origin));}
  for(const event of eventsOf(sample)){
   const source=Number(event.source),target=Number(event.target),edge=key(source,target);
   steps++; seenNodes.add(source);seenNodes.add(target);reconstructed.add(edge);counted(edgeCounts,edge);counted(nodeCounts,target);counted(targetCounts,target);
   if(event.kind!=="legal_correct")errors++;
   if(event.kind==="illegal")illegal++;
   if(event.kind==="legal_mismatch")mismatch++;
   if(trueEdges.has(edge))recovered.add(edge);else {fake.add(edge);fakeEvents++;}
   if(event.kind==="legal_correct"&&trueEdges.has(edge)){solid.add(edge);counted(solidCounts,edge);}
  }
  const reference=sample.reference_nodes||[];
  reference.forEach(n=>counted(referenceNodes,Number(n)));
  for(let i=1;i<reference.length;i++)counted(referenceEdges,key(reference[i-1],reference[i]));
 }
 const retainedSeen=[...seenNodes].filter(n=>trueNodes.has(n)).length;
 return {samples:selectedSamples.length,steps,errors,illegal,mismatch,fakeEvents,reconstructed,seenNodes,fake,recovered,solid,edgeCounts,solidCounts,nodeCounts,targetCounts,originCounts,referenceEdges,referenceNodes,precision:ratio(recovered.size,reconstructed.size),recall:ratio(recovered.size,trueEdges.size),nodeCoverage:ratio(retainedSeen,trueNodes.size),targetNodeCoverage:ratio([...targetCounts.keys()].filter(n=>trueNodes.has(n)).length,trueNodes.size),solidRecall:ratio(solid.size,trueEdges.size)};
}

function chooseSamples(){
 const value=$("sample-subset").value.trim(), status=$("subset-status");
 status.className="status";
 if(!value){status.textContent="Numbers are 1-based positions within the selected cohort.";return pool.slice(0,Number($("sample-count").value));}
 const found = new Set(), invalid=[], byId=new Map(pool.map((s,i)=>[idOf.get(s),i]));
 for(const raw of value.split(/[,;\n]+/)){
  const token=raw.trim();if(!token)continue;
  if(byId.has(token)){found.add(byId.get(token));continue;}
  const match=token.match(/^(\d+)\s*[-–:]\s*(\d+)$/);
  if(match){const a=Number(match[1]),b=Number(match[2]);if(a<1||b<a||b>pool.length){invalid.push(token);continue;}for(let i=a;i<=b;i++)found.add(i-1);}
  else if(/^\d+$/.test(token)&&Number(token)>=1&&Number(token)<=pool.length){found.add(Number(token)-1);}
  else invalid.push(token);
 }
 status.textContent=`Explicit selection: ${found.size} sample${found.size===1?"":"s"}; cumulative slider is ignored.`+(invalid.length?` Unrecognized / out of range: ${invalid.join(", ")}`:"");
 if(invalid.length)status.classList.add("error");
 return [...found].sort((a,b)=>a-b).map(i=>pool[i]);
}

function metric(label,value,help){return `<div class="card metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="help">${esc(help)}</div></div>`;}
function update(){
 selected=chooseSamples();currentAnalysis=analyze(selected);
 const a=currentAnalysis;
 $("range-value").textContent=`${$("sample-count").value} / ${pool.length}`;
 $("count-description").textContent=$("sample-count").value;
 $("selection-badge").textContent=`${selected.length} selected`;
 $("selection-metrics").innerHTML=[
  metric("DIRECTION STEPS",number(a.steps,0),`${number(a.samples,0)} selected samples`),
  metric("ERROR EVENTS / SAMPLE",number(ratio(a.errors,a.samples)),`${a.illegal} illegal · ${a.mismatch} location mismatches`),
  metric("FAKE TRAVERSALS / SAMPLE",number(ratio(a.fakeEvents,a.samples)),`Repeats included · ${a.fake.size} distinct fake edges`),
  metric("EDGE PRECISION",pct(a.precision),`${a.recovered.size} real / ${a.reconstructed.size} reconstructed`),
  metric("TRUE-EDGE RECALL",pct(a.recall),`${a.recovered.size} / ${trueEdges.size} true edges recovered`),
  metric("PROBED NODE COVERAGE",pct(a.targetNodeCoverage),`${pct(a.nodeCoverage)} including prompt origins`)
 ].join("");
 $("recon-map-subtitle").textContent=`${a.reconstructed.size} unique topology edges · ${a.fake.size} fake · ${a.errors} action/location errors`;
 renderMaps();renderChart();renderSampleChoices();renderFrequency();
}

function svgFrame(){
 const width=620,height=570,pad=42,spacing=Math.min((width-2*pad)/Math.max(cols-1,1),(height-2*pad)/Math.max(rows-1,1));
 const spanX=spacing*(cols-1),spanY=spacing*(rows-1),left=(width-spanX)/2,top=(height-spanY)/2;
 const point=n=>[left+((Number(n)-1)%cols)*spacing,top+Math.floor((Number(n)-1)/cols)*spacing];
 return {width,height,point,spacing,r:Math.max(2.2,Math.min(7,spacing*.10))};
}
const frame=svgFrame();
function titleText(text){return `<title>${esc(text)}</title>`;}
function nodeTitle(n,a){return `Node ${n} · row ${Math.floor((n-1)/cols)+1}, column ${(n-1)%cols+1} · training visits: ${graph.node_counts?.[String(n)]||0} · post-action probe predictions: ${a.targetCounts.get(n)||0} · supplied prompt origins: ${a.originCounts.get(n)||0} · reference visits: ${a.referenceNodes.get(n)||0}`;}
function edgePath(source,target,offset=0){
 const [x1,y1]=frame.point(source),[x2,y2]=frame.point(target);
 if(source===target){const d=13+Math.abs(offset);return `M ${x1-2} ${y1-3} C ${x1-d*2} ${y1-d*2.5},${x1+d*2} ${y1-d*2.5},${x1+2} ${y1-3}`;}
 const length=Math.hypot(x2-x1,y2-y1),mx=(x1+x2)/2-(y2-y1)/length*offset,my=(y1+y2)/2+(x2-x1)/length*offset;
 return offset?`M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`:`M ${x1} ${y1} L ${x2} ${y2}`;
}
function svgStart(label){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="${esc(label)}"><rect width="100%" height="100%" fill="#ffffff"/>`;}
function gridBackground(){
 if(rows*cols>2500)return "";
 const out=[];
 for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
  const n=r*cols+c+1;if(trueNodes.has(n))continue;const [x,y]=frame.point(n);
  out.push(`<circle cx="${x}" cy="${y}" r="1.2" fill="#e8edf3"/>`);
 }
 return out.join("");
}
function graphNodes(a,reconstructed=false){
 const labels=$("show-labels").checked,out=[];
 const drawNodes=new Set([...nodes,...(reconstructed?a.seenNodes:[])]);
 for(const n of drawNodes){
  const [x,y]=frame.point(n),observed=a.seenNodes.has(n);
  const fill=reconstructed?(observed?"#244f80":"#e7edf4"):"#315779";
  const opacity=reconstructed&&!observed?.7:1;
  const label=labels?`<text x="${x+frame.r+3}" y="${y-frame.r-2}" font-size="${Math.max(7,Math.min(10,frame.spacing*.19))}" fill="${reconstructed&&!observed?"#a1aebb":"#37516d"}" paint-order="stroke" stroke="white" stroke-width="2.8" stroke-linejoin="round">${n}</text>`:"";
  out.push(`<g class="interactive" data-tip="${esc(nodeTitle(n,a))}" opacity="${opacity}">${titleText(nodeTitle(n,a))}<circle class="graph-node" cx="${x}" cy="${y}" r="${frame.r}" fill="${fill}" stroke="white" stroke-width="1.5"/>${label}</g>`);
 }
 return out.join("");
}
function renderMaps(){
 const a=currentAnalysis,weighted=$("show-frequency").checked;
 let left=svgStart("True training graph")+gridBackground();
 for(const [u,v] of edges){const count=Number(graph.edge_counts?.[key(u,v)]||0),tip=`True edge ${u} ↔ ${v} · training traversals: ${count} · selected reconstructed traversals: ${a.edgeCounts.get(key(u,v))||0} · selected reference traversals: ${a.referenceEdges.get(key(u,v))||0}`;
  left+=`<path class="interactive" data-tip="${esc(tip)}" d="${edgePath(u,v)}" fill="none" stroke="#7995b3" stroke-opacity=".69" stroke-width="${weighted?Math.min(6,1.2+Math.log1p(count)*.63):1.7}">${titleText(tip)}</path>`;}
 left+=graphNodes(a)+"</svg>";$("true-map").innerHTML=left;
 let right=svgStart("Reconstructed graph from selected samples")+gridBackground();
 if($("show-missing").checked)for(const [u,v] of edges){if(a.recovered.has(key(u,v)))continue;right+=`<path d="${edgePath(u,v)}" fill="none" stroke="#dce4ed" stroke-width="1.2">${titleText(`Missing true edge ${u} ↔ ${v}`)}</path>`;}
 const threshold=Math.max(0,Math.min(1,Number($("min-confidence").value)||0));
 const enabled={legal_correct:$("show-correct").checked,legal_mismatch:$("show-mismatch").checked,illegal:$("show-illegal").checked}, aggregates=new Map();
 let displayed=0;
 for(const sample of selected)for(const event of eventsOf(sample)){
  if(!enabled[event.kind]||Number(event.probe_confidence)<threshold)continue;
  const k=[event.source,event.direction,event.target,event.kind].join("|");
  if(!aggregates.has(k))aggregates.set(k,{...event,count:0,confidences:[],probabilities:[]});
  const item=aggregates.get(k);item.count++;item.confidences.push(Number(event.probe_confidence));item.probabilities.push(Number(event.direction_probability));displayed++;
 }
 const groups=new Map();for(const event of aggregates.values()){const k=key(event.source,event.target);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(event);}
 const drawing=[];
 for(const group of groups.values()){
  group.sort((a,b)=>String(a.kind).localeCompare(String(b.kind))||Number(a.source)-Number(b.source)||String(a.direction).localeCompare(String(b.direction)));
  group.forEach((e,index)=>{
   let offset=0;
   if(group.length>1)offset=(index-(group.length-1)/2)*Math.min(12,frame.spacing*.18);
   if(e.kind!=="legal_correct"&&Math.abs(offset)<1)offset=Math.min(15,frame.spacing*.23);
   // Canonical orientation makes opposite directions separate consistently.
   if(Number(e.source)>Number(e.target))offset=-offset;
   const isTrue=trueEdges.has(key(e.source,e.target)),tip=`${e.source} —${e.direction}→ ${e.target} · ${kindNames[e.kind]} · ${e.count} occurrence${e.count===1?"":"s"} · probe confidence mean ${pct(mean(e.confidences))}, min ${pct(e.confidences.reduce((a,b)=>Math.min(a,b),Infinity))}, max ${pct(e.confidences.reduce((a,b)=>Math.max(a,b),-Infinity))} · P(direction) mean ${pct(mean(e.probabilities))} · topology ${isTrue?"exists":"FAKE"} · training traversals ${graph.edge_counts?.[key(e.source,e.target)]||0}`;
   drawing.push({order:e.kind==="legal_correct"?0:1,html:`<path class="interactive" data-tip="${esc(tip)}" d="${edgePath(Number(e.source),Number(e.target),offset)}" fill="none" stroke="${kindColors[e.kind]}" stroke-opacity="${e.kind==="legal_correct"?.74:.84}" stroke-width="${weighted?Math.min(7,1.3+Math.log1p(e.count)*.8):1.9}" ${e.kind!=="legal_correct"?'stroke-dasharray="3 4" stroke-linecap="round"':""}>${titleText(tip)}</path>`});
  });
 }
 drawing.sort((a,b)=>a.order-b.order).forEach(item=>{right+=item.html;});right+=graphNodes(a,true)+"</svg>";$("recon-map").innerHTML=right;
 $("recon-footer").textContent=`${displayed} / ${a.steps} events visible, aggregated into ${aggregates.size} source–direction–target/type records. Edge thickness reflects occurrences${weighted?"":" when enabled"}.`;
 for(const host of [$("true-map"),$("recon-map")])host.querySelectorAll("[data-tip]").forEach(element=>{element.addEventListener("pointerenter",()=>{$("hover-readout").textContent=element.dataset.tip;});});
}

function renderSampleChoices(){
 const previous=$("detail-sample").value;
 $("detail-sample").innerHTML=selected.length?selected.map(sample=>`<option value="${esc(idOf.get(sample))}">${esc(idOf.get(sample))} · ${sample.origin} → ${sample.destination} · ${esc(sample.cohort)}</option>`).join(""):"<option>No sample selected</option>";
 if(selected.some(s=>idOf.get(s)===previous))$("detail-sample").value=previous;
 renderSample();
}
function renderSample(){
 const sample=selected.find(s=>idOf.get(s)===$("detail-sample").value);
 if(!sample){$("sample-meta").textContent="Use the slider or explicit sample selector to inspect a generated route.";$("step-body").innerHTML='<tr><td colspan="9" class="empty">No sample selected.</td></tr>';return;}
 const termination=typeof sample.termination==="object"?JSON.stringify(sample.termination):sample.termination;
 const reference=sample.reference_directions||[];
 $("sample-meta").textContent=`ID ${idOf.get(sample)} · origin ${sample.origin} → requested destination ${sample.destination} · ${eventsOf(sample).length} generated directions · termination: ${termination??"unspecified"} · physical route valid: ${sample.physical_valid??"unspecified"} · physical destination reached: ${sample.destination_reached??"unspecified"} · probe destination reached: ${sample.probe_destination_reached??"unspecified"} · exact training-route match: ${sample.exact_training_route_match??"unspecified"}. Legal reference: ${reference.join(" ")||"(empty)"}.`;
 $("step-body").innerHTML=eventsOf(sample).map(e=>{
  const alternatives=(e.probe_topk||[]).map(p=>`${p.node}: ${pct(p.probability)}`).join(" · ");
  return `<tr><td>${esc(e.step)}</td><td>${esc(e.source)}</td><td><b>${esc(e.direction)}</b></td><td>${e.expected_target==null?"—":esc(e.expected_target)}</td><td><b>${esc(e.target)}</b></td><td><span class="state ${e.kind==="legal_correct"?"good":e.kind==="legal_mismatch"?"orange":"red"}">${esc(kindNames[e.kind]||e.kind)}</span></td><td>${pct(e.probe_confidence)}</td><td>${pct(e.direction_probability)}</td><td class="wrappable">${esc(alternatives)}</td></tr>`;
 }).join("")||'<tr><td colspan="9" class="empty">This sample generated no direction events. Check its termination reason.</td></tr>';
}

function renderFrequency(){
 const a=currentAnalysis;
 if(frequencyMode==="edges"){
  frequencyRows=edges.map(([u,v])=>({id:key(u,v),train:Number(graph.edge_counts?.[key(u,v)]||0),reconstructed:a.edgeCounts.get(key(u,v))||0,solid:a.solidCounts.get(key(u,v))||0,reference:a.referenceEdges.get(key(u,v))||0}));
 }else{
  frequencyRows=nodes.map(n=>({id:String(n),train:Number(graph.node_counts?.[String(n)]||0),reconstructed:a.targetCounts.get(n)||0,prompt_origins:a.originCounts.get(n)||0,reference:a.referenceNodes.get(n)||0}));
 }
 const sort=$("frequency-sort").value;
 frequencyRows.sort((a,b)=>{
  if(sort==="missing")return Number(b.reconstructed===0)-Number(a.reconstructed===0)||b.train-a.train;
  if(sort==="rare")return a.train-b.train||b.reconstructed-a.reconstructed;
  if(sort==="generated")return b.reconstructed-a.reconstructed||b.train-a.train;
  return b.train-a.train||a.reconstructed-b.reconstructed;
 });
 const filtered=$("only-missing").checked?frequencyRows.filter(r=>!r.reconstructed):frequencyRows;
 const headers=frequencyMode==="edges"?["True edge","Training traversals","Reconstructed traversals","Solid traversals","Reference traversals","Coverage"]:["True node","Training visits","Post-action probe visits","Prompt origins","Reference visits","Coverage"];
 $("frequency-head").innerHTML=`<tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr>`;
 $("frequency-body").innerHTML=filtered.map(row=>`<tr><td><b>${esc(row.id)}</b></td><td>${row.train}</td><td>${row.reconstructed}</td><td>${frequencyMode==="edges"?row.solid:row.prompt_origins}</td><td>${row.reference}</td><td><span class="state ${row.reconstructed?"good":"missed"}">${row.reconstructed?"Recovered":"Missing"}</span></td></tr>`).join("")||`<tr><td class="empty" colspan="${headers.length}">No ${$("only-missing").checked?"missing ":""}${frequencyMode}.</td></tr>`;
}

function prepareCoverage(){
 const recovered=new Set(),fake=new Set(),visited=new Set(),points=[{n:0,r:0,v:0,f:0}];
 for(let i=0;i<pool.length;i++){
  const sample=pool[i];visited.add(Number(sample.origin));
  for(const e of eventsOf(sample)){const k=key(e.source,e.target);(trueEdges.has(k)?recovered:fake).add(k);visited.add(Number(e.source));visited.add(Number(e.target));}
  points.push({n:i+1,r:ratio(recovered.size,trueEdges.size)||0,v:ratio([...visited].filter(n=>trueNodes.has(n)).length,trueNodes.size)||0,f:ratio(fake.size,trueEdges.size)||0});
 }
 coveragePoints=points;
}
function renderChart(){
 const width=600,height=190,pad={l:40,r:14,t:12,b:26},span=width-pad.l-pad.r,plotHeight=height-pad.t-pad.b,points=coveragePoints;
 const max=points.reduce((m,p)=>Math.max(m,p.f),1),x=n=>pad.l+span*n/Math.max(pool.length,1),y=v=>pad.t+plotHeight*(1-v/max);
 let chart=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Cumulative true edge recall, node coverage and fake edges relative to true edges">`;
 for(let i=0;i<=4;i++){const value=max*i/4;chart+=`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(value)}" y2="${y(value)}" stroke="#e7edf4"/><text x="${pad.l-7}" y="${y(value)+4}" text-anchor="end" font-size="10" fill="#7e8b9b">${Math.round(value*100)}%</text>`;}
 for(const [field,color] of [["r","#176f93"],["v","#7c68b0"],["f","#ca4264"]])chart+=`<path d="${points.map((p,i)=>`${i?"L":"M"} ${x(p.n)} ${y(p[field])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2"/>`;
 if(!$("sample-subset").value.trim()){const n=Number($("sample-count").value);chart+=`<line x1="${x(n)}" x2="${x(n)}" y1="${pad.t}" y2="${height-pad.b}" stroke="#516b88" stroke-dasharray="3 4"/>`;}
 chart+=`<text x="${pad.l}" y="${height-7}" font-size="10" fill="#7e8b9b">0</text><text x="${width-pad.r}" y="${height-7}" text-anchor="end" font-size="10" fill="#7e8b9b">${pool.length} samples</text></svg>`;
 $("coverage-chart").innerHTML=chart;
}

function displayValue(value){
 if(typeof value==="number")return number(value,5);
 if(value===null||value===undefined)return "—";
 if(typeof value==="object")return JSON.stringify(value);
 return String(value);
}
function humanize(key){return key.replace(/_/g," ").replace(/\b\w/,c=>c.toUpperCase());}
function fullMetrics(){
 const metrics=data.metrics||{};
 $("full-metrics").innerHTML=["all","seen","unseen"].map(cohort=>{
  const result=metrics[cohort]||{},summary=result.summary||result;
  const entries=Object.entries(summary).filter(([k,v])=>v===null||typeof v!=="object");
  return `<div><h3>${cohort==="all"?"Both cohorts":cohort==="seen"?"Seen endpoint pairs":"Unseen endpoint pairs"}</h3><div class="scroll"><table><tbody>${entries.map(([k,v])=>`<tr><td>${esc(humanize(k))}</td><td>${esc(displayValue(v))}</td></tr>`).join("")||'<tr><td class="empty">No saved metrics.</td></tr>'}</tbody></table></div></div>`;
 }).join("");
 const datasetMetadata=Object.fromEntries(Object.entries(data.dataset||{}).filter(([k])=>k!=="graph"));
 const details=[
  ["Configuration",data.config||{}],
  ["Model training",data.training||{}],
  ["Probe training and held-out validation",data.probe||{}],
  ["Dataset and split diagnostics",datasetMetadata],
  ["Metric definitions",metrics.all?.definitions||{}],
  ["Rarity groups and structural diagnostics",Object.fromEntries(["all","seen","unseen"].map(k=>[k,{rarity:metrics[k]?.rarity,topology_distortion:metrics[k]?.topology_distortion}]))]
 ];
 $("experiment-details").innerHTML=details.map(([title,value])=>`<details><summary>${esc(title)}</summary><div><pre>${esc(JSON.stringify(value,null,2))}</pre></div></details>`).join("");
}

function download(name,content,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvCell(value){let text=typeof value==="object"&&value!==null?JSON.stringify(value):String(value??"");if(/^[=+@]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';}
function csv(columns,rows){return columns.map(csvCell).join(",")+"\r\n"+rows.map(row=>columns.map(c=>csvCell(row[c])).join(",")).join("\r\n");}
function resetCohort(){
 const cohort=$("cohort").value;pool=cohort==="all"?samples:samples.filter(s=>s.cohort===cohort);
 $("sample-count").max=pool.length;$("sample-count").value=pool.length;$("sample-subset").value="";
 $("cohort-status").textContent=`${pool.length} generated sample${pool.length===1?"":"s"}`;prepareCoverage();update();
}
$("cohort").addEventListener("change",resetCohort);
$("sample-count").addEventListener("input",update);
$("sample-subset").addEventListener("input",update);
$("clear-subset").addEventListener("click",()=>{$("sample-subset").value="";update();});
for(const id of ["show-correct","show-mismatch","show-illegal","show-missing","show-labels","show-frequency","min-confidence"])$(id).addEventListener("input",renderMaps);
$("detail-sample").addEventListener("change",renderSample);
for(const mode of ["edges","nodes"])$("frequency-"+mode).addEventListener("click",()=>{frequencyMode=mode;$("frequency-edges").classList.toggle("active",mode==="edges");$("frequency-nodes").classList.toggle("active",mode==="nodes");renderFrequency();});
$("frequency-sort").addEventListener("change",renderFrequency);$("only-missing").addEventListener("change",renderFrequency);
$("export-events").addEventListener("click",()=>{const records=selected.flatMap(s=>eventsOf(s).map(e=>({sample_id:idOf.get(s),cohort:s.cohort,origin:s.origin,destination:s.destination,...e})));const columns=["sample_id","cohort","origin","destination","step","source","direction","target","expected_target","kind","probe_confidence","direction_probability","probe_topk"];download("selected-events.csv",csv(columns,records),"text/csv;charset=utf-8");});
$("export-frequency").addEventListener("click",()=>{const columns=frequencyMode==="edges"?["id","train","reconstructed","solid","reference"]:["id","train","reconstructed","prompt_origins","reference"];download(`selected-${frequencyMode}-frequency.csv`,csv(columns,frequencyRows),"text/csv;charset=utf-8");});
$("export-json").addEventListener("click",()=>download("experiment.json",JSON.stringify(data,null,2),"application/json;charset=utf-8"));
$("true-map-subtitle").textContent=`${nodes.length} visited nodes · ${edges.length} undirected edges`;
$("grid-size").textContent=`${rows} × ${cols}`;
$("probe-diagnostic").textContent=`Location-probe accuracy on legal reference walks: validation ${pct(data.probe?.validation?.accuracy)}, seen-pair test ${pct(data.probe?.test_seen?.accuracy)}, unseen-pair test ${pct(data.probe?.test_unseen?.accuracy)}. Interpret the reconstructed map alongside these scores; accuracy after illegal prefixes is unvalidated.`;
fullMetrics();resetCohort();
})();
</script>
</body>
</html>
'''
