// Dataset controls are kept separate from the graph editor.
let datasetInfo = null;
function datasetControls() {
  if (!document.querySelector('#dataPanel')) {
    const panel = document.createElement('div'); panel.id = 'dataPanel'; panel.className = 'panel'; panel.style.marginBottom = '20px';
    panel.innerHTML = `<div class="panel-head"><h2>데이터셋 · CSV 업로드 및 열 선택</h2><small>LOCAL DATA ONLY</small></div><div class="section"><div class="fields"><div><label class="label" for="dataSource">데이터 소스</label><select data-dataset id="dataSource"><option value="synthetic">합성 데이터 (실행 점검)</option><option value="csv">업로드한 CSV</option></select></div><div><label class="label" for="csvFile">CSV 파일 · 최대 50MB / 200,000행</label><input data-dataset type="file" accept=".csv,.tsv,text/csv" id="csvFile"></div></div><p class="helper">UTF-8·CP949, 쉼표·탭·세미콜론 지원. 업로드한 원본은 이 컴퓨터의 uploads 폴더에 저장됩니다.</p><button class="btn" id="demoCsv" data-dataset style="margin-top:10px">인공 CSV 예제로 체험</button><div id="dataStatus" role="status" style="margin-top:12px;font-size:12px"></div><div id="datasetFields"></div></div>`;
    document.querySelector('#builderview').prepend(panel);
    document.querySelector('#dataSource').onchange = () => {config.data_source = document.querySelector('#dataSource').value;if(config.data_source==='synthetic'){config.features=12;config.sequence=8;document.querySelector('#features').value=12;document.querySelector('#sequence').value=8;}datasetControls();refresh()};
    document.querySelector('#csvFile').onchange = uploadDataset;
    document.querySelector('#demoCsv').onclick = () => {const lines=['session,bytes,packets,label'];for(let g=0;g<20;g++)for(const label of ['normal','attack'])for(let j=0;j<3;j++)lines.push(`s${g},${10+g+j+(label==='attack'?100:0)},${j+1},${label}`);uploadDataset(new File([lines.join('\n')],'artificial-upload-demo.csv',{type:'text/csv'}));};
  }
  document.querySelector('#dataSource').value = config.data_source || 'synthetic';
  const real = config.data_source === 'csv';
  document.querySelector('#features').disabled = real;document.querySelector('#sequence').disabled = real;
  if (real) {config.sequence=1;config.features=Math.max(1,config.feature_columns.length);document.querySelector('#sequence').value=1;document.querySelector('#features').value=config.features;}
  document.querySelector('#datasetFields').classList.toggle('hidden', !real);
  document.querySelector('#dataAside').textContent = real ? '데이터 · 업로드한 CSV / Flow 단위' : '데이터 · 합성 Flow 시퀀스';
}
async function uploadDataset(fileOverride) {
  const file=fileOverride instanceof File?fileOverride:document.querySelector('#csvFile').files[0];if(!file)return;
  const status=document.querySelector('#dataStatus');status.textContent='CSV 업로드 및 전체 열 검사 중…';
  if(file.size>50*1024*1024){status.textContent='CSV는 최대 50MB까지 지원합니다.';return}
  document.querySelector('#csvFile').disabled=true;
  try {
    const response=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'text/csv','X-Filename':encodeURIComponent(file.name)},body:file});
    const info=await response.json();if(!response.ok)throw Error(info.error);
    datasetInfo=info;config.dataset_id=info.id;config.data_source='csv';config.feature_columns=[];config.group_column='';
    const label=info.columns.find(c=>/^(label|class|attack|category|target)$/i.test(c.name));
    config.label_column=label?.name||'';config.feature_columns=info.columns.filter(c=>c.numeric&&c.name!==config.label_column).slice(0,64).map(c=>c.name);
    drawDataset();datasetControls();await refresh();
  } catch(e) {status.textContent='업로드 실패: '+e.message;toast(e.message)}
  finally{document.querySelector('#csvFile').disabled=false}
}
function drawDataset() {
  if(!datasetInfo)return;
  const info=datasetInfo;
  document.querySelector('#dataStatus').textContent=`${info.name} · ${info.rows.toLocaleString()}행 / ${info.columns.length}열 · 원본 SHA-256 ${info.sha256.slice(0,12)}…`;
  const options='<option value="">열을 선택하세요</option>'+info.columns.map(c=>`<option value="${esc(c.name)}">${esc(c.name)} (${c.unique}종류)</option>`).join('');
  document.querySelector('#datasetFields').innerHTML=`<div class="fields" style="margin-top:20px;grid-template-columns:1fr 1fr 1fr"><div><label class="label" for="labelColumn">라벨 열</label><select data-dataset id="labelColumn">${options}</select></div><div><label class="label" for="splitMode">분할 방식 · 목표 70 / 15 / 15</label><select data-dataset id="splitMode"><option value="stratified">클래스 비율 유지 · 무작위 행 분할</option><option value="group">그룹 분리 · 세션 중복 방지</option></select></div><div><label class="label" for="groupColumn">세션 / 그룹 열 (그룹 분할 시)</label><select data-dataset id="groupColumn">${options}</select></div></div><p class="helper">CSV는 한 행 = 하나의 Flow, 시퀀스 길이 1입니다. 클래스별 최소 7행, 2~50개 클래스가 필요합니다. 결측·무한대·문자 특성은 자동 보정하지 않습니다. 무작위 행 분할에는 세션 중복 가능성이 있으므로 세션 열이 있으면 그룹 분리를 사용하세요.</p><div id="labelCounts" style="margin-top:12px"></div><span class="label" style="margin-top:18px">입력 특성 선택 · 숫자 열만 / 라벨·그룹 자동 제외</span><div id="featureChecks" style="display:flex;flex-wrap:wrap;gap:6px 18px"></div><details style="margin-top:15px"><summary style="font-size:12px;cursor:pointer">처음 5행 미리보기</summary><div style="overflow:auto"><table><thead><tr>${info.columns.map(c=>`<th>${esc(c.name)}</th>`).join('')}</tr></thead><tbody>${info.preview.map(r=>`<tr>${r.map(v=>`<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`;
  document.querySelector('#labelColumn').value=config.label_column;
  document.querySelector('#splitMode').value=config.split_mode;
  document.querySelector('#groupColumn').value=config.group_column;
  for(const [id,key] of [['labelColumn','label_column'],['splitMode','split_mode'],['groupColumn','group_column']])document.querySelector('#'+id).onchange=()=>{config[key]=document.querySelector('#'+id).value;filterFeatures();drawFeatureChecks();refresh()};
  filterFeatures();drawFeatureChecks();
}
function filterFeatures(){config.feature_columns=config.feature_columns.filter(c=>c!==config.label_column&&(config.split_mode!=='group'||c!==config.group_column));}
function drawFeatureChecks(){
  const info=datasetInfo;if(!info)return;
  document.querySelector('#groupColumn').disabled=config.split_mode!=='group';
  const label=info.columns.find(c=>c.name===config.label_column);
  document.querySelector('#labelCounts').innerHTML=label?.counts?Object.entries(label.counts).map(([k,n])=>`<span class="selected-metric">${esc(k||'(빈 라벨)')} · ${n}</span>`).join(''):'<span class="helper">라벨 열을 선택하세요. 클래스가 50개를 초과하는 열은 라벨로 사용할 수 없습니다.</span>';
  document.querySelector('#featureChecks').innerHTML=info.columns.map((c,i)=>{const blocked=!c.numeric||c.name===config.label_column||(config.split_mode==='group'&&c.name===config.group_column);return `<label class="check"><input data-dataset data-feature-index="${i}" type="checkbox" ${blocked?'disabled':''} ${config.feature_columns.includes(c.name)&&!blocked?'checked':''}>${esc(c.name)}${!c.numeric?` <span class="helper">(비숫자/결측 ${c.invalid_numeric})</span>`:''}</label>`}).join('');
  document.querySelectorAll('[data-feature-index]').forEach(e=>e.onchange=()=>{config.feature_columns=Array.from(document.querySelectorAll('[data-feature-index]:checked')).map(e=>info.columns[Number(e.dataset.featureIndex)].name);datasetControls();refresh()});
  datasetControls();
}
async function restoreDataset(){if(config.dataset_id){try{datasetInfo=await api('/api/datasets/'+config.dataset_id);drawDataset();datasetControls();await refresh()}catch(e){document.querySelector('#dataStatus').textContent=e.message}}}
function datasetResult(r){if(!r.dataset)return '';const d=r.dataset;return `<div class="panel" style="margin:18px 0"><div class="panel-head"><h2>데이터셋 및 분할 기록</h2><small>UPLOADED CSV</small></div><div class="section"><b>${esc(d.name)}</b> · ${d.rows.toLocaleString()}행 · 라벨 ${esc(d.label)}<p class="helper">특성: ${d.features.map(esc).join(', ')}<br>분할: ${esc(d.split_mode)} ${d.group_column?' / '+esc(d.group_column):''} · seed ${d.seed}<br>정규화는 학습 데이터만으로 계산했습니다. ${d.split_mode==='stratified'?'무작위 행 분할은 같은 세션의 중복을 배제하지 않습니다.':'그룹은 분할 사이에서 중복되지 않습니다. 시간 순서 일반화 평가는 포함하지 않습니다.'}<br>SHA-256: ${esc(d.sha256)}</p></div><div style="overflow:auto"><table><thead><tr><th>CLASS</th><th>TRAIN</th><th>VALIDATION</th><th>TEST</th></tr></thead><tbody>${d.classes.map(c=>`<tr><td>${esc(c)}</td><td>${d.distribution.train[c]}</td><td>${d.distribution.val[c]}</td><td>${d.distribution.test[c]}</td></tr>`).join('')}</tbody></table></div></div>`}
