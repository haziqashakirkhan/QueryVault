const $=s=>document.querySelector(s);
const state={view:'workspace',tech:'ZERO-SHOT',busy:false,answer:null,q:'',error:false,failNext:false,
docs:[]};
const stages=['Uploading','Processing','Generating embeddings','Indexing'];
const CH=[
['RAG.pdf','Page 12',.91,'Retrieval-Augmented Generation combines a retriever with a generator. At query time the question is embedded, the nearest chunks are fetched from a vector store, and they are supplied to the language model as grounding context, so the answer reflects the source documents rather than only the model\'s training data.'],
['Embeddings.pdf','Page 7',.86,'An embedding maps text to a dense vector so that passages with similar meaning sit close together. Cosine similarity between a query vector and chunk vectors is the standard measure used to rank candidates in semantic search, replacing exact keyword overlap.'],
['Transformers.pdf','Page 23',.78,'Attention lets the generator weigh every retrieved token against the question. When context is placed ahead of the question in the prompt, the model can condition its output on the supplied passages and cite them in the final response.']];
const SCORES=[['ZERO-SHOT',3.6,3.9,3.7,'Direct, but sometimes drifts beyond the retrieved text.'],['FEW-SHOT',4.2,4.1,4.3,'Worked examples fix the answer format and improve precision.'],['ROLE-BASED',4.0,4.5,4.4,'A researcher persona gives the clearest, most focused prose.']];
const styleTxt={'ZERO-SHOT':'','FEW-SHOT':'','ROLE-BASED':''};

function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function renderDocs(){
  const ul=$('#docs');
  ul.innerHTML=state.docs.map(d=>{const ok=d.stage==='Indexed';return `<li class="${ok?'':'busy'}"><div class="n">${esc(d.name)}</div><div class="m"><span><span class="dot"></span>${d.stage}${ok?'':'…'}</span><span>${d.type}${ok&&d.pages?' · '+d.pages+' pp · '+d.chunks+' chunks':''}</span></div>${ok?'':`<div class="bar"><i style="width:${d.p*25}%"></i></div>`}</li>`}).join('')||'<li style="border:0;color:#aebaa8;font-size:13px">No documents yet.</li>';
  const n=state.docs.filter(d=>d.stage==='Indexed').length;
  $('#count').textContent=n+' DOCUMENT'+(n===1?'':'S')+' INDEXED';
}
function pipe(i){const s=['Question','Retrieval','Context','Answer'];return `<div class="pipe" aria-label="Pipeline">${s.map((t,k)=>`<span class="${k<i?'done':k===i?'on':''}">${t}</span>${k<3?'<hr>':''}`).join('')}</div>`}
function renderMain(){
  const m=$('#main');
  if(state.view==='evaluation'){
    const N={zero_shot:'ZERO-SHOT',few_shot:'FEW-SHOT',role_based:'ROLE-BASED'},ev=state.ev;
    const head=`<div class="label">Prompt engineering</div><h1>Evaluation</h1>`;
    const runBtn=`<button class="go" id="runEv" style="padding:14px 24px;margin-top:22px" ${state.evRun?'disabled':''}>${state.evRun?'Running evaluation…':ev?'Run again':'Run evaluation'}</button>`;
    const err=state.evErr?`<div class="err" role="alert" style="margin-top:20px"><b>Evaluation failed</b>${esc(state.evErr)}</div>`:'';
    const status=state.evRun?`<div class="status" role="status" style="margin-top:22px"><span class="pulse"></span>Answering 5 questions with 3 techniques and scoring each. This can take a minute or two.</div>`:'';
    let body='';
    if(ev===undefined)body='<p class="lede">Loading the latest results…</p>';
    else if(!ev)body='<p class="lede">No evaluation has been run yet. It asks the same five questions with each prompting technique and scores every answer from 1 to 5.</p>';
    else{
      const T=Object.keys(ev.summary),best=k=>Math.max(...T.map(t=>ev.summary[t][k]??0));
      const cell=(t,k)=>{const v=ev.summary[t][k];return v==null?'<td>-</td>':`<td class="${v===best(k)?'best':''}"><div class="score">${v.toFixed(2)}<i><u style="width:${v*20}%"></u></i></div></td>`};
      body=`<p class="lede">Same retrieval and same five questions for every technique, scored 1 to 5 by a judge model (${esc(ev.judge_model||'')}). Answers by ${esc(ev.llm_model||'')}.</p>
      <div class="scroll"><table><thead><tr><th>Technique</th><th>Accuracy</th><th>Clarity</th><th>Relevance</th><th>Overall</th><th>Scored</th></tr></thead><tbody>${T.map(t=>`<tr><td>${N[t]||t}${ev.winner===t?' (best)':''}</td>${['accuracy','clarity','relevance','overall'].map(k=>cell(t,k)).join('')}<td>${ev.summary[t].scored_questions}/${ev.questions.length}</td></tr>`).join('')}</tbody></table></div>
      ${ev.analysis?`<p class="note">${esc(ev.analysis)}</p>`:''}
      <h2 style="font:400 24px var(--serif);color:var(--forest);margin:44px 0 6px">By question</h2>
      ${ev.questions.map((q,i)=>`<details><summary>${i+1}. ${esc(q.question)}</summary>${Object.keys(q.results).map(t=>{const r=q.results[t];return `<div class="qa"><b>${N[t]||t}</b> ${r.scores?`<span>Accuracy ${r.scores.accuracy} · Clarity ${r.scores.clarity} · Relevance ${r.scores.relevance}</span>`:''}<p>${r.error?'Error: '+esc(r.error):esc(r.answer||'(empty answer)')}</p>${r.reason?`<small>${esc(r.reason)}</small>`:''}</div>`}).join('')}</details>`).join('')}
      <p class="note" style="font-size:14px">Run on ${esc((ev.created_at||'').replace('T',' ').slice(0,16))} UTC. Five questions is a small sample, so treat differences as indicative.</p>`;
    }
    m.innerHTML=`<div class="wrap eval">${head}${body}${runBtn}${status}${err}</div>`;
    const b=$('#runEv');if(b)b.onclick=runEval;return}
  if(state.view==='settings'){
    const c=state.cfg||{};
    const rows=[['Language model',c.llm_model],['Embedding model',c.embedding_model],['Chunk size',c.chunk_size],['Chunk overlap',c.chunk_overlap],['Retrieved chunks',c.top_k],['Chunks indexed',c.chunk_count],['Language model key',c.llm_ready?'Configured':'Missing: add GROQ_API_KEY to .env']];
    m.innerHTML=`<div class="wrap"><div class="label">Configuration</div><h1>Settings</h1><p class="lede">Current pipeline configuration, read from the server.</p><div class="set">${rows.map(r=>`<label><span>${r[0]}</span><span>${esc(String(r[1]??'-'))}</span></label>`).join('')}</div></div>`;return}
  const ready=state.docs.some(d=>d.stage==='Indexed');
  if(!state.docs.length){
    m.innerHTML=`<div class="empty"><h1>Build your knowledge base.</h1><p class="lede" style="margin:0 auto">Upload documents to begin asking questions with semantic retrieval.</p><button class="go" id="up2">Upload documents</button></div>`;
    $('#up2').onclick=()=>$('#file').click();return}
  const a=state.answer,i=state.busy?state.step:a?3:0;
  let res='';
  if(state.busy)res=`<div class="result" role="status"><div class="status"><span class="pulse"></span>${['Reading your question…','Retrieving context from the vector index…','Ranking the top three chunks…','Generating a grounded answer…'][state.step]}</div></div>`;
  else if(state.error)res=`<div class="result"><div class="err" role="alert"><b>Question failed</b>${esc(state.errMsg||'No answer was generated.')}<br><button id="retry">Retry question</button></div></div>`;
  else if(a)res=`<div class="result"><div class="label">AI response</div><p class="q">${esc(a.q)}</p><div class="answer">${a.html}</div><div class="ground">GROUNDED IN DOCUMENT CONTEXT</div><div class="meta"><span>Model: ${esc(state.model||'')}</span><span>${a.t}s</span><span>${a.chunks.length} sources retrieved</span><span>${state.tech.toLowerCase()}</span></div></div>`;
  m.innerHTML=`<div class="wrap"><div class="label">Document intelligence</div><h1>Ask your documents.</h1><p class="lede">Find precise answers using semantic retrieval and AI-generated responses grounded in your knowledge base.</p>${pipe(i)}
  <div class="ask"><textarea id="q" rows="1" aria-label="Question" placeholder="Ask a question about your documents..." ${state.busy?'disabled':''}>${esc(state.q)}</textarea><button class="go" id="go" ${state.busy||!ready?'disabled':''}>Ask</button></div>
  <fieldset class="tech"><legend>Prompt technique</legend>${['ZERO-SHOT','FEW-SHOT','ROLE-BASED'].map(t=>`<label><input type="radio" name="t" value="${t}" ${state.tech===t?'checked':''}><span>${t}</span></label>`).join('')}</fieldset>${res}</div>`;
  const q=$('#q');q.oninput=()=>{state.q=q.value;q.style.height='auto';q.style.height=q.scrollHeight+'px'};
  q.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}};
  $('#go').onclick=ask;
  m.querySelectorAll('[name=t]').forEach(r=>r.onchange=()=>{state.tech=r.value;renderMain()});
  const rt=$('#retry');if(rt)rt.onclick=ask;
}
function renderChunks(){
  const c=$('#chunks');
  if(!state.docs.length){c.innerHTML='<p style="margin-top:22px;font-size:13px;color:var(--gray)">Retrieved passages appear here after you ask a question.</p>';return}
  if(state.busy&&state.step<2){c.innerHTML='<div style="margin-top:22px">'+'<div class="skel"></div><div class="skel" style="width:70%"></div>'.repeat(3)+'</div>';return}
  if(!state.answer){c.innerHTML='<p style="margin-top:22px;font-size:13px;color:var(--gray)">Retrieved passages appear here after you ask a question.</p>';return}
  c.innerHTML=state.answer.chunks.map((h,k)=>`<div class="chunk" id="c${k}"><button class="ch" aria-expanded="false"><span class="no">0${k+1}</span><span class="t"><b>${h[0]}</b><small>${h[1]}</small></span><span class="sim">Similarity ${h[2].toFixed(2)}<i><u style="width:${h[2]*100}%"></u></i></span></button><p class="pv">${esc(h[3])}</p><button class="more">Show full chunk</button></div>`).join('');
  c.querySelectorAll('.chunk').forEach(el=>{const t=()=>{const o=el.classList.toggle('open');el.querySelector('.ch').setAttribute('aria-expanded',o);el.querySelector('.more').textContent=o?'Collapse':'Show full chunk'};el.querySelector('.ch').onclick=t;el.querySelector('.more').onclick=t});
}
function normalize(q,d,t){
  const src=d.sources||d.chunks||d.contexts||d.context||[];
  const chunks=src.slice(0,3).map(s=>[s.source||s.name||s.document||s.filename||'Unknown',s.page!=null?'Page '+s.page:(s.chunk_index!=null?'Chunk '+s.chunk_index:''),Number(s.score??s.similarity??0),s.text||s.content||s.chunk||'']);
  const html=String(d.answer||d.response||'').split(/\n{2,}/).map(x=>'<p>'+esc(x.trim()).replace(/\n/g,'<br>')+'</p>').join('');
  return {q,t,html,chunks};
}
async function ask(){
  if(state.busy)return;
  const q=state.q.trim();if(!q){$('#q')&&$('#q').focus();return}
  state.busy=true;state.error=false;state.answer=null;state.step=0;render();
  const start=Date.now();
  const t1=setTimeout(()=>{state.step=1;render()},400),t2=setTimeout(()=>{state.step=3;render()},1800);
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,technique:state.tech.toLowerCase().replace('-','_')})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'Request failed ('+r.status+').');
    state.answer=normalize(q,d,((Date.now()-start)/1000).toFixed(1));
  }catch(e){state.error=true;state.errMsg=e.message}
  clearTimeout(t1);clearTimeout(t2);state.busy=false;render();
  if(state.answer&&matchMedia('(max-width:1100px)').matches)setCtx(false);
}
async function loadEval(){
  try{const r=await fetch('/api/evaluate/latest');state.ev=r.ok?await r.json():null}catch(e){state.ev=null}
  render();
}
async function runEval(){
  if(state.evRun)return;
  state.evRun=true;state.evErr=null;render();
  try{
    const r=await fetch('/api/evaluate',{method:'POST'});
    const d=await r.json().catch(()=>({}));
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'Request failed ('+r.status+').');
    state.ev=d;
  }catch(e){state.evErr=e.message}
  state.evRun=false;render();
}
async function load(){
  try{
    const [c,dl]=await Promise.all([fetch('/api/config').then(r=>r.json()),fetch('/api/documents').then(r=>r.json())]);
    state.cfg=c;state.model=c.llm_model;
    state.docs=dl.map(d=>({name:d.name,type:d.name.split('.').pop().toUpperCase(),pages:0,chunks:d.chunks,stage:'Indexed',p:4}));
  }catch(e){state.model='unavailable'}
  render();
}
function setCtx(o){$('#ctx').classList.toggle('open',o);$('#ctxT').setAttribute('aria-expanded',o);$('#ctxI').textContent=o?'Hide':'Show'}
$('#ctxT').onclick=()=>setCtx(!$('#ctx').classList.contains('open'));
function render(){renderMain();renderChunks();renderDocs();$('#ctx').classList.toggle('hide',state.view!=='workspace')}
function setView(v){state.view=v;if(v==='evaluation'&&state.ev===undefined)loadEval();document.querySelectorAll('nav [data-view]').forEach(b=>b.toggleAttribute('aria-current',b.dataset.view===v)||b.removeAttribute('aria-current'));document.querySelectorAll('nav [data-view]').forEach(b=>{if(b.dataset.view===v)b.setAttribute('aria-current','page')});$('#ws').style.gridTemplateColumns=v==='workspace'?'':'288px minmax(0,1fr)';closeDrawer();render()}
document.querySelectorAll('nav [data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
/* drawer */
function closeDrawer(){$('#side').classList.remove('open');$('#menu').setAttribute('aria-expanded',false);const s=$('.scrim');s&&s.remove()}
$('#menu').onclick=()=>{const o=$('#side').classList.toggle('open');$('#menu').setAttribute('aria-expanded',o);if(o){const s=document.createElement('div');s.className='scrim';s.onclick=closeDrawer;document.body.append(s)}else closeDrawer()};
addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawer()});
/* upload */
async function addFiles(files){
  const fd=new FormData(),rows=[];
  [...files].forEach(f=>{fd.append('files',f);const d={name:f.name,type:f.name.split('.').pop().toUpperCase(),pages:0,chunks:0,stage:'Uploading',p:1};rows.push(d);state.docs.push(d)});
  render();
  const t=setTimeout(()=>{rows.forEach(d=>{d.stage='Generating embeddings';d.p=3});renderDocs()},1200);
  try{
    const r=await fetch('/api/documents',{method:'POST',body:fd});
    const res=await r.json().catch(()=>({}));
    if(!r.ok)throw new Error(typeof res.detail==='string'?res.detail:'Upload failed.');
    state.docs=state.docs.filter(d=>!rows.includes(d)&&!res.saved.some(s=>s.name===d.name));
    res.saved.forEach(s=>state.docs.push({name:s.name,type:s.name.split('.').pop().toUpperCase(),pages:0,chunks:s.chunks,stage:'Indexed',p:4}));
    if(res.errors.length)alert(res.errors.map(e=>e.name+': '+e.error).join('\n'));
  }catch(e){state.docs=state.docs.filter(d=>!rows.includes(d));alert(e.message)}
  clearTimeout(t);render();
}
$('#up1').onclick=()=>$('#file').click();
$('#file').onchange=e=>{addFiles(e.target.files);e.target.value=''};
const dz=$('#drop');
['dragenter','dragover'].forEach(v=>dz.addEventListener(v,e=>{e.preventDefault();dz.classList.add('over')}));
['dragleave','drop'].forEach(v=>dz.addEventListener(v,e=>{e.preventDefault();dz.classList.remove('over')}));
dz.addEventListener('drop',e=>addFiles(e.dataTransfer.files));
dz.tabIndex=0;dz.onkeydown=e=>{if(e.key==='Enter')$('#file').click()};dz.onclick=()=>$('#file').click();
render();load();