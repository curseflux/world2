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
<p>Every arrow is a probe-decoded response to a forced direction. Illegal actions always receive a node prediction; an arrow alone does not establish a believed edge.</p>
<section>
<label>Contexts <select id="kind"><option value="reference">Held-out valid prefixes</option><option value="generated_valid">Generated valid prefixes</option><option value="generated_invalid">Generated invalid prefixes</option></select></label>
<label>Direction <select id="direction"></select></label>
<label>Source <input id="source" type="number" min="1" placeholder="All nodes"></label>
<label>Min agreement <input id="agreement" type="number" min="0" max="1" step="0.05" value="0"></label>
<label>Min contexts <input id="count" type="number" min="1" value="1"></label>
<label><input id="matched" type="checkbox">Only prefixes with correct pre-action probe</label>
<p><span class="blue">Blue: legal and modal target correct.</span> <span class="orange">Orange: legal, modal target wrong.</span> <span class="red">Red: forced illegal direction.</span> Thickness shows agreement across contexts. Arrowheads show direction of transition.</p>
<div id="stats"></div></section>
<div class="maps"><section><h2>True map</h2><div id="truth"></div></section><section><h2>Modal decoded transitions</h2><div id="inferred"></div></section></div>
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
$('direction').innerHTML=dirs.map(a=>'<option>'+a+'</option>').join('');
$('details').textContent=JSON.stringify({settings:data.settings,coverage:data.coverage,provenance:data.provenance,implementation:data.implementation,interpretation:data.interpretation},null,2);
function point(n){return [35+((n-1)%graph.cols)*430/Math.max(1,graph.cols-1),35+Math.floor((n-1)/graph.cols)*430/Math.max(1,graph.rows-1)];}
function edge(u,v,color,width,dotted,arrow){let [x,y]=point(u),[a,b]=point(v);let shape;
if(u===v)shape=`<circle cx="${x}" cy="${y-8}" r="8" fill="none"`;
else{const d=Math.hypot(a-x,b-y),dx=(a-x)/d*7,dy=(b-y)/d*7;shape=`<line x1="${x+dx}" y1="${y+dy}" x2="${a-dx}" y2="${b-dy}"`;}
return shape+` stroke="${color}" stroke-width="${width}" ${dotted?'stroke-dasharray="4 3"':''} ${arrow?'marker-end="url(#arrow)"':''}/>`;}
function drawing(lines){return '<svg viewBox="0 0 500 500"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#526174"/></marker></defs>'+lines+graph.nodes.map(n=>{const [x,y]=point(n);return `<circle cx="${x}" cy="${y}" r="4" fill="#334963"/><text x="${x+5}" y="${y-5}" font-size="9">${n}</text>`;}).join('')+'</svg>';}
$('truth').innerHTML=drawing(graph.edges.map(([u,v])=>edge(u,v,'#9cacc0',1.5,false,false)).join(''));
function filtered(){return ( $('matched').checked?data.source_correct_transitions:data.transitions).filter(r=>r.kind===$('kind').value&&r.direction===$('direction').value&&(!$('source').value||r.source===Number($('source').value))&&r.agreement>=Number($('agreement').value)&&r.contexts>=Number($('count').value));}
function update(){const rows=filtered();$('stats').textContent=rows.length+' transitions shown. '+data.contexts.length+' prefixes tested; '+data.events.length+' action/inverse branches.';
$('inferred').innerHTML=drawing(rows.map(r=>edge(r.source,r.target,r.true_neighbor==null?'#b32946':r.target===r.true_neighbor?'#1c768b':'#b16b05',1+4*r.agreement,r.target!==r.true_neighbor,true)).join(''));
$('rows').innerHTML=rows.map((r,i)=>`<tr class="pick" data-i="${i}"><td>${r.source} / ${esc(r.direction)}</td><td>${esc(r.true_neighbor)}</td><td>${r.target}</td><td>${r.contexts}</td><td>${pct(r.agreement)}</td><td>${pct(r.mean_direction_probability)}</td><td>${pct(r.cycle_return_to_source_rate)}</td><td>${esc(r.independent_reverse_target)} (${pct(r.independent_reverse_agreement)})</td></tr>`).join('');
document.querySelectorAll('tr.pick').forEach(e=>e.onclick=()=>{chosen=rows[Number(e.dataset.i)];$('selected').textContent=`Source ${chosen.source}, forced ${chosen.direction}: votes ${JSON.stringify(chosen.votes)}. Independent reverse starts at the modal target in separate contexts.`;details();});chosen=null;$('selected').textContent='Select a transition above, or enter a route ID. Without a selected row, branches for all directions are shown.';details();}
function details(){const route=$('route').value.trim(),step=Number($('step').value);const events=data.events.filter(e=>{const c=contexts.get(e.context_id);return c.kind===$('kind').value&&(!chosen||c.source===chosen.source&&e.direction===chosen.direction)&&(!route||c.route_id===route)&&(!step||c.step===step)&&(!$('matched').checked||c.known_source!=null&&c.pre_node===c.known_source);});
$('branches').innerHTML=events.slice(0,1000).map(e=>{const c=contexts.get(e.context_id);return `<tr><td>${esc(c.route_id)} / ${c.step}</td><td>${c.source}${c.known_source==null?' (inferred)':''}</td><td>${c.pre_node} (${pct(c.pre_confidence)})</td><td>${esc(e.direction)}</td><td>${e.target}</td><td>${pct(e.confidence)}</td><td>${pct(e.direction_probability)}</td><td>${esc(e.inverse_direction)} → ${e.return_node} (${pct(e.return_confidence)})</td><td>${c.destination} / ${esc(c.prefix.join(' '))}</td></tr>`;}).join('');}
for(const id of ['kind','direction','source','agreement','count','matched'])$(id).onchange=update;
for(const id of ['route','step'])$(id).oninput=details;update();
</script></html>'''
