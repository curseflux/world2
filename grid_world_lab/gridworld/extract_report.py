"""Offline viewer for context-dependent, directed probe transitions."""
import json


def render_extraction(payload, path):
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    path.write_text(TEMPLATE.replace('__PAYLOAD__', encoded), encoding='utf-8')


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forced-action map extraction</title>
<style>
body{font:15px/1.5 system-ui;margin:24px;background:#f5f7fa;color:#182536}h1{font-size:26px}
section{background:white;border:1px solid #d4dce7;padding:16px;margin:14px 0;border-radius:8px}
.maps{display:grid;grid-template-columns:1fr 1fr;gap:16px}.maps section{min-width:0}svg{width:100%;max-height:540px}
label{display:inline-block;margin:6px 14px 6px 0}input,select{font:inherit;max-width:150px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:8px;text-align:left;border-bottom:1px solid #dde4ed}th{position:sticky;top:0;background:#eef3fa}
tr.pick{cursor:pointer}tr.pick:hover{background:#eef6ff}.scroll{max-height:420px;overflow:auto}pre{white-space:pre-wrap}
.red{color:#b32946}.orange{color:#b16b05}.blue{color:#1c768b}small{color:#506078}
@media(max-width:800px){body{margin:12px}.maps{grid-template-columns:1fr}}
</style>
<h1>Forced-action map extraction</h1>
<p>The model-coordinate map places node labels by fitting repeated directional beliefs. It does not use row-major node coordinates. Forced illegal actions always receive a probe prediction, so consistency and residuals matter more than a single arrow.</p>
<section>
<label>Transition table contexts <select id="kind"><option value="reference">Held-out valid prefixes</option><option value="generated_valid">Generated valid prefixes</option><option value="generated_invalid">Generated invalid prefixes</option></select></label>
<label>Direction <select id="direction"><option value="ALL">All directions</option></select></label>
<label>Source <input id="source" type="number" min="1" placeholder="All nodes"></label>
<label>Min agreement <input id="agreement" type="number" min="0" max="1" step="0.05" value="0"></label>
<label>Min contexts <input id="count" type="number" min="1" value="1"></label>
<label><input id="matched" type="checkbox" checked>Only prefixes with correct pre-action probe</label>
<label><input id="directionMarks" type="checkbox">Show one-way direction marks</label>
<label><input id="conflicts" type="checkbox">Show conflicting constraints</label>
<p><span class="blue">Blue: independently reciprocal.</span> Gray: one-way or reverse unavailable. Reciprocal pairs are collapsed into one line. Optional red dashed lines conflict with the fitted geometry. Filters hide edges and table rows; they do not refit node coordinates.</p>
<div id="stats"></div></section>
<div class="maps"><section><h2>True map</h2><small>Node placement uses actual row-major coordinates.</small><div id="truth"></div></section><section><h2>Model-coordinate map</h2><small>Node placement is inferred jointly from all accepted source/action → target constraints. Disconnected components are translated apart for display.</small><div id="inferred"></div></section></div>
<section><h2>Transitions across contexts</h2><small>Click a row to inspect its individual branches. Modal agreement is a vote fraction, not probe confidence. A tie uses the smaller node ID.</small><div class="scroll"><table><thead><tr><th>Source/action</th><th>True neighbor</th><th>Modal target</th><th>Contexts</th><th>Agreement</th><th>Mean P(action)</th><th>Inverse return</th><th>Independent reverse</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>Branch details</h2><label>Route ID <input id="route" type="text" placeholder="e.g. seen-1"></label><label>Step <input id="step" type="number" min="1" placeholder="All steps"></label><p id="selected">Select a transition above, or enter a route ID.</p><small>Up to 1,000 matching branches shown; JSON and CSV include all branches.</small><div class="scroll"><table><thead><tr><th>Route / step</th><th>Source</th><th>Pre-probe</th><th>Forced action</th><th>Decoded target</th><th>Confidence</th><th>P(action)</th><th>After inverse</th><th>Destination / prefix</th></tr></thead><tbody id="branches"></tbody></table></div></section>
<section><details><summary>Coverage, settings, provenance and interpretation</summary><pre id="details"></pre></details></section>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id), graph=data.graph;
const contexts=new Map(data.contexts.map(c=>[c.id,c]));let chosen=null;
const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
const dirs=graph.directions||['N','NE','E','SE','S','SW','W','NW'];
$('direction').innerHTML+=dirs.map(a=>'<option>'+a+'</option>').join('');
$('details').textContent=JSON.stringify({settings:data.settings,coverage:data.coverage,model_geometry:{kind:data.model_geometry.kind,source_requires_correct_preprobe:data.model_geometry.source_requires_correct_preprobe,constraint_count:data.model_geometry.constraint_count,component_count:data.model_geometry.component_count,weighted_rmse:data.model_geometry.weighted_rmse,weighted_near_exact_rate:data.model_geometry.weighted_near_exact_rate,independent_reciprocal_rate:data.model_geometry.independent_reciprocal_rate},provenance:data.provenance,implementation:data.implementation,interpretation:data.interpretation},null,2);
const trueSpan=Math.max(graph.cols-1,graph.rows-1,1),trueScale=430/trueSpan;
const truePoint=n=>[35+(430-(graph.cols-1)*trueScale)/2+((n-1)%graph.cols)*trueScale,35+(430-(graph.rows-1)*trueScale)/2+Math.floor((n-1)/graph.cols)*trueScale];
const modelPositions=new Map(data.model_geometry.positions.map(p=>[p.node,p]));
const gx=data.model_geometry.positions.map(p=>p.x),gy=data.model_geometry.positions.map(p=>p.y),gminx=Math.min(...gx),gmaxx=Math.max(...gx),gminy=Math.min(...gy),gmaxy=Math.max(...gy);
const modelSpan=Math.max(gmaxx-gminx,gmaxy-gminy,1),modelScale=430/modelSpan,modelOffX=35+(430-(gmaxx-gminx)*modelScale)/2,modelOffY=35+(430-(gmaxy-gminy)*modelScale)/2;
const modelPoint=n=>{const p=modelPositions.get(n);return [modelOffX+(p.x-gminx)*modelScale,modelOffY+(p.y-gminy)*modelScale];};
function lineEdge(point,u,v,color,width,dotted,tip=''){let [x,y]=point(u),[a,b]=point(v);let shape;
if(u===v||Math.hypot(a-x,b-y)<1)shape=`<circle cx="${x}" cy="${y-7}" r="6" fill="none"`;
else{const d=Math.hypot(a-x,b-y),dx=(a-x)/d*8,dy=(b-y)/d*8;shape=`<line x1="${x+dx}" y1="${y+dy}" x2="${a-dx}" y2="${b-dy}"`;}
return shape+` stroke="${color}" stroke-width="${width}" stroke-opacity=".68" ${dotted?'stroke-dasharray="3 3"':''}>${tip?`<title>${esc(tip)}</title>`:''}</${shape.startsWith('<circle')?'circle':'line'}>`;}
function directionMark(point,u,v,color){if(u===v)return '';const [x,y]=point(u),[a,b]=point(v),d=Math.hypot(a-x,b-y);if(d<1)return '';const ux=(a-x)/d,uy=(b-y)/d,vx=-uy,vy=ux,mx=(x+a)/2,my=(y+b)/2;return `<path d="M ${mx-ux*4+vx*3} ${my-uy*4+vy*3} L ${mx+ux*4} ${my+uy*4} L ${mx-ux*4-vx*3} ${my-uy*4-vy*3} Z" fill="${color}" fill-opacity=".78"/>`;}
function drawing(point,nodes,lines,compass=''){const dense=nodes.length>64,r=dense?8:7,font=dense?7.5:9;return '<svg viewBox="0 0 500 500" role="img" aria-label="Graph map">'+lines+nodes.map(n=>{const [x,y]=point(n);return `<g><title>Node ${n}</title><circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="#334963" stroke-width="1.5"/><text x="${x}" y="${y+font*.36}" text-anchor="middle" font-size="${font}" font-weight="600" fill="#22364d">${n}</text></g>`;}).join('')+compass+'</svg>';}
$('truth').innerHTML=drawing(truePoint,graph.nodes,graph.edges.map(([u,v])=>lineEdge(truePoint,u,v,'#9cacc0',1.2,false)).join(''));
function filtered(){const direction=$('direction').value;return ( $('matched').checked?data.source_correct_transitions:data.transitions).filter(r=>r.kind===$('kind').value&&(direction==='ALL'||r.direction===direction)&&(!$('source').value||r.source===Number($('source').value))&&r.agreement>=Number($('agreement').value)&&r.contexts>=Number($('count').value));}
function update(){const rows=filtered(),direction=$('direction').value,source=$('source').value;const constraints=data.model_geometry.constraints.filter(e=>(direction==='ALL'||e.direction===direction)&&(!source||e.source===Number(source))&&e.agreement>=Number($('agreement').value)&&e.contexts>=Number($('count').value));
$('stats').textContent=`${rows.length} transition rows shown. The fixed model geometry uses ${data.model_geometry.constraint_count} ${data.model_geometry.kind} constraints${data.model_geometry.source_requires_correct_preprobe?' with a correct pre-action probe':''}, in ${data.model_geometry.component_count} components; weighted residual RMSE ${data.model_geometry.weighted_rmse==null?'—':data.model_geometry.weighted_rmse.toFixed(3)}; ${pct(data.model_geometry.weighted_near_exact_rate)} near-exact constraint weight.`;
const seen=new Set(),display=[];for(const e of constraints){const conflict=e.residual>.35;if(conflict&&!$('conflicts').checked)continue;const reciprocal=!conflict&&e.independent_reciprocal&&e.source!==e.target;const key=reciprocal?[Math.min(e.source,e.target),Math.max(e.source,e.target)].join(':'):`${e.source}:${e.direction}:${e.target}`;if(seen.has(key))continue;seen.add(key);display.push({...e,reciprocal});}
const modelLines=display.map(e=>{const conflict=e.residual>.35,color=conflict?'#b32946':e.reciprocal?'#1c768b':'#64748b',tip=e.reciprocal?`${e.source} ↔ ${e.target}; reciprocal directional belief; agreement ${pct(e.agreement)}`:`${e.source} ${e.direction}→ ${e.target}; agreement ${pct(e.agreement)}; residual ${e.residual.toFixed(3)}`;return lineEdge(modelPoint,e.source,e.target,color,conflict?1.25:1.1+.7*e.agreement,conflict,tip)+($('directionMarks').checked&&!e.reciprocal&&!conflict?directionMark(modelPoint,e.source,e.target,color):'');}).join('');
$('inferred').innerHTML=drawing(modelPoint,graph.nodes,modelLines,'<text x="470" y="25" text-anchor="middle" font-size="11">N ↑</text><text x="470" y="42" text-anchor="middle" font-size="11">W ← · → E</text><text x="470" y="59" text-anchor="middle" font-size="11">S ↓</text>');
$('rows').innerHTML=rows.map((r,i)=>`<tr class="pick" data-i="${i}"><td>${r.source} / ${esc(r.direction)}</td><td>${esc(r.true_neighbor)}</td><td>${r.target}</td><td>${r.contexts}</td><td>${pct(r.agreement)}</td><td>${pct(r.mean_direction_probability)}</td><td>${pct(r.cycle_return_to_source_rate)}</td><td>${esc(r.independent_reverse_target)} (${pct(r.independent_reverse_agreement)})</td></tr>`).join('');
document.querySelectorAll('tr.pick').forEach(e=>e.onclick=()=>{chosen=rows[Number(e.dataset.i)];$('selected').textContent=`Source ${chosen.source}, forced ${chosen.direction}: votes ${JSON.stringify(chosen.votes)}. Independent reverse starts at the modal target in separate contexts.`;details();});chosen=null;$('selected').textContent='Select a transition above, or enter a route ID. Without a selected row, branches for all directions are shown.';details();}
function details(){const route=$('route').value.trim(),step=Number($('step').value);const events=data.events.filter(e=>{const c=contexts.get(e.context_id);return c.kind===$('kind').value&&(!chosen||c.source===chosen.source&&e.direction===chosen.direction)&&(!route||c.route_id===route)&&(!step||c.step===step)&&(!$('matched').checked||c.known_source!=null&&c.pre_node===c.known_source);});
$('branches').innerHTML=events.slice(0,1000).map(e=>{const c=contexts.get(e.context_id);return `<tr><td>${esc(c.route_id)} / ${c.step}</td><td>${c.source}${c.known_source==null?' (inferred)':''}</td><td>${c.pre_node} (${pct(c.pre_confidence)})</td><td>${esc(e.direction)}</td><td>${e.target}</td><td>${pct(e.confidence)}</td><td>${pct(e.direction_probability)}</td><td>${esc(e.inverse_direction)} → ${e.return_node} (${pct(e.return_confidence)})</td><td>${c.destination} / ${esc(c.prefix.join(' '))}</td></tr>`;}).join('');}
for(const id of ['kind','direction','source','agreement','count','matched','directionMarks','conflicts'])$(id).onchange=update;
for(const id of ['route','step'])$(id).oninput=details;update();
</script></html>'''
