let comparison = null, compareTimer = null;
const presets = [
  {name:'CNN',blocks:['conv'],head:'dense',pool:true},
  {name:'LSTM',blocks:['lstm'],head:'dense',pool:true},
  {name:'Attention',blocks:['attention'],head:'dense',pool:true},
  {name:'CNN + LSTM + Attention',blocks:['conv','lstm','attention'],head:'dense',pool:true},
  {name:'CNN + MoE',blocks:['conv'],head:'moe',pool:true},
  {name:'LSTM + MoE',blocks:['lstm'],head:'moe',pool:true},
  {name:'Attention + MoE',blocks:['attention'],head:'moe',pool:true},
  {name:'CNN + LSTM + Attention + MoE',blocks:['conv','lstm','attention'],head:'moe',pool:true}
];
let variants = presets.map(v=>({...v,blocks:[...v.blocks]}));
function initComparisons(){
  const host=$('#compareview');
  host.innerHTML=`<div class="panel"><div class="panel-head"><h2>실험 비교 설계</h2><small>SAME DATA · SAME SPLIT · SEQUENTIAL RUNS</small></div><div class="section"><p class="run-config">구조도 화면에서 선택한 데이터셋·특성·라벨·분할·seed·optimizer·epoch를 공통으로 사용합니다. CSV는 한 번 분할한 파일을 모든 모델에 복사합니다. F1-macro, 모든 클래스의 재현율, p50/p95/p99 지연을 항상 측정합니다.</p><div id="variantList" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px"></div><div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap"><button class="btn" id="addCurrent">＋ 현재 구조도 조합 추가</button><button class="btn primary" id="compareRun">▷ 선택한 조합 비교 실행</button><button class="btn" id="compareExport" disabled>↓ 비교표 CSV</button></div><p class="helper">2~8개 조합을 선택하세요. 실행 순서는 아래 표의 순서입니다. CPU·2 threads·batch 1·warmup 30·200회 측정. CSV는 시퀀스 길이 1이므로 시계열 모델의 장기 의존성 성능을 검증하지 않습니다.</p></div></div><div class="panel" style="margin-top:18px"><div class="panel-head"><h2>비교 기록</h2><select id="comparisonHistory" aria-label="비교 기록 선택" style="width:auto;max-width:65%"><option value="">기록 선택</option></select></div><div id="comparisonResults"><div class="empty"><b>같은 조건에서 모델을 비교하세요</b>비교할 조합을 고르고 실행하면 결과가 순서대로 채워집니다.</div></div></div>`;
  drawVariants();
  $('#compareRun').onclick=startComparison;
  $('#addCurrent').onclick=()=>{read();const v={name:`현재 조합 ${variants.length+1}`,blocks:[...config.blocks],head:config.head,pool:config.pool};if(variants.length>=16)return toast('조합 목록은 최대 16개입니다.');variants.push(v);drawVariants()};
  $('#compareExport').onclick=exportComparison;
  $('#comparisonHistory').onchange=()=>{if($('#comparisonHistory').value)watchComparison($('#comparisonHistory').value)};
  loadComparisonHistory(true);
}
function drawVariants(){
  const old=Array.from(document.querySelectorAll('[data-variant]:checked')).map(e=>Number(e.dataset.variant));
  $('#variantList').innerHTML=variants.map((v,i)=>`<label class="block-option" style="margin:0;justify-content:flex-start;gap:10px"><input type="checkbox" data-variant="${i}" ${old.length?old.includes(i)?'checked':'':i<4||i===7?'checked':''}><div><strong>${esc(v.name)}</strong><small>${esc(v.blocks.join(' → ')||'projection')} → ${esc(v.head)}${v.pool?'':' · pooling 없음'}</small></div></label>`).join('');
}
async function loadComparisonHistory(restore=false){try{
  const list=await api('/api/comparisons');const selected=comparison?.id;
  $('#comparisonHistory').innerHTML='<option value="">기록 선택</option>'+list.map(b=>`<option value="${b.id}">${esc(new Date(b.started*1000).toLocaleString('ko-KR'))} · ${esc(b.name)} · ${b.count}개 · ${esc(b.status)}</option>`).join('');
  if(selected)$('#comparisonHistory').value=selected;
  if(restore){const id=localStorage.getItem('assembly-comparison');const active=list.find(b=>['running','queued'].includes(b.status));if(active||list.some(b=>b.id===id))watchComparison(active?.id||id)}
}catch(e){toast('비교 기록: '+e.message)}}
async function startComparison(){
  read();const chosen=Array.from(document.querySelectorAll('[data-variant]:checked')).map(e=>variants[Number(e.dataset.variant)]);
  if(chosen.length<2||chosen.length>8)return toast('조합을 2~8개 선택하세요.');
  $('#compareRun').disabled=true;
  try{const {id}=await api('/api/compare',{base:structuredClone(config),variants:chosen});localStorage.setItem('assembly-comparison',id);await watchComparison(id);loadComparisonHistory()}
  catch(e){$('#compareRun').disabled=false;toast(e.message)}
}
const comparisonStatus={queued:'대기',running:'실행 중',completed:'완료',failed:'실패',partial:'일부 실패',interrupted:'중단'};
async function watchComparison(id){
  clearTimeout(compareTimer);
  try{comparison=await api('/api/comparisons/'+id);localStorage.setItem('assembly-comparison',id);renderComparison();const active=['queued','running'].includes(comparison.status);$('#compareRun').disabled=active;$('#compareBadge').textContent=comparisonStatus[comparison.status];if(active)compareTimer=setTimeout(()=>watchComparison(id),1400);else loadComparisonHistory()}
  catch(e){$('#compareRun').disabled=false;toast(e.message);$('#comparisonResults').textContent='비교 기록을 읽지 못했습니다. 기록을 다시 선택해 재연결할 수 있습니다.'}
}
function renderComparison(){
  const b=comparison, jobs=b.jobs, complete=jobs.filter(j=>j.status==='completed'), classes=complete[0]?.result.classes||b.base.classes||['Benign','Scan','Flood'];
  $('#compareExport').disabled=!complete.length;
  const best=complete.length?Math.max(...complete.map(j=>j.result.f1_macro)):null;
  const fastest=complete.length?Math.min(...complete.map(j=>j.result.latency_ms.p95)):null;
  const th=s=>`<th style="white-space:nowrap">${esc(s)}</th>`;
  $('#comparisonResults').innerHTML=`<div class="section"><div style="display:flex;justify-content:space-between;gap:10px;align-items:center"><b>${complete.length} / ${jobs.length} 완료 · ${esc(comparisonStatus[b.status])}</b><span class="badge">${b.base.data_source==='csv'?'UPLOADED CSV':'SYNTHETIC SMOKE TEST'}</span></div><p class="run-config" style="margin:12px 0 0">${esc(b.dataset?.name||'데이터 준비 중')} · seed ${b.base.seed} · ${esc(b.base.optimizer)} · ${b.base.epochs} epochs · CPU / 2 threads<br>특성: ${esc(b.base.data_source==='csv'?b.base.feature_columns.join(', '):b.base.features+'개')} · 라벨: ${esc(b.base.label_column||'합성 3 classes')} · 분할: ${esc(b.dataset?.split_mode||b.base.split_mode)} ${b.base.split_mode==='group'?'/ '+esc(b.base.group_column):''}<br>공통 데이터/분할 지문: <code>${esc(b.fingerprint||'준비 중')}</code></p>${b.error?`<p style="color:#b64f4f">${esc(b.error)}</p>`:''}</div><div style="overflow:auto"><table><thead><tr>${['MODEL / STATUS','F1-macro ↑',...classes.map(c=>'Recall · '+c+' ↑'),'p50 ms ↓','p95 ms ↓','p99 ms ↓','Parameters','DETAIL'].map(th).join('')}</tr></thead><tbody>${jobs.map(j=>{const r=j.result,ok=j.status==='completed'&&r;return `<tr><td style="min-width:190px"><strong>${esc(j.config.name)}</strong><div class="helper">${esc(comparisonStatus[j.status])} · ${esc(j.config.blocks.join(' → ')||'projection')} → ${esc(j.config.head)}</div></td><td style="${ok&&r.f1_macro===best?'color:var(--teal);font-weight:700':''}">${ok?r.f1_macro.toFixed(4):'—'}</td>${classes.map(c=>`<td>${ok&&r.recall[c]!==undefined?(r.recall[c]*100).toFixed(1)+'%':'—'}</td>`).join('')}<td>${ok?r.latency_ms.p50.toFixed(3):'—'}</td><td style="${ok&&r.latency_ms.p95===fastest?'color:var(--blue);font-weight:700':''}">${ok?r.latency_ms.p95.toFixed(3):'—'}</td><td>${ok?r.latency_ms.p99.toFixed(3):'—'}</td><td>${ok?r.parameters.toLocaleString():'—'}</td><td><button class="btn" data-compare-detail="${j.id}" style="white-space:nowrap">상세</button></td></tr>`}).join('')}</tbody></table></div><div class="section"><span class="helper">초록색: 현재 완료된 실험 중 최고 테스트 F1 · 파란색: 최저 p95 (동률 포함). 테스트 표를 반복 튜닝에 사용하면 선택 편향이 생길 수 있습니다. 최종 모델은 별도 보류 데이터로 확인하세요. 순차 측정으로 실험 간 CPU 경쟁을 피하지만 시스템 부하와 실행 순서의 영향은 남습니다.</span></div><div id="comparisonDetail"></div>`;
  document.querySelectorAll('[data-compare-detail]').forEach(e=>e.onclick=()=>{const j=jobs.find(j=>j.id===e.dataset.compareDetail);$('#comparisonDetail').innerHTML=`<div class="section"><b>${esc(j.config.name)} · ${esc(j.id)}</b><p class="helper">${esc(j.error?.message||j.stage||'')}<br>저장 위치: runs/${esc(j.id)}</p><details open><summary>설정 및 실행 로그</summary><pre style="margin-top:12px;max-height:350px">${esc(JSON.stringify(j.config,null,2))}\n${esc(j.log||'실행 로그 대기 중')}</pre></details></div>`});
}
function exportComparison(){
  if(!comparison)return;const b=comparison,classes=b.jobs.find(j=>j.result)?.result.classes||[];
  const header=['comparison_id','model','status','blocks','head','seed','optimizer','epochs','data_fingerprint','F1_macro',...classes.map(c=>'recall_'+c),'p50_ms','p95_ms','p99_ms','parameters','error'];
  const rows=b.jobs.map(j=>{const r=j.status==='completed'?j.result:null;return[b.id,j.config.name,j.status,j.config.blocks.join(' → '),j.config.head,j.config.seed,j.config.optimizer,j.config.epochs,b.fingerprint||'',r?.f1_macro??'',...classes.map(c=>r?.recall[c]??''),r?.latency_ms.p50??'',r?.latency_ms.p95??'',r?.latency_ms.p99??'',r?.parameters??'',j.error?.message||'']});
  const cell=v=>{let s=String(v);if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};
  const text='\ufeff'+[header,...rows].map(r=>r.map(cell).join(',')).join('\r\n');const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`comparison-${b.id}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
