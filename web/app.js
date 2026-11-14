"use strict";
const $ = id => document.getElementById(id);
const app = {api:null, config:{}, labels:[], history:[], offset:0, pageSize:25, count:0, runs:[], busy:false, polling:false, replay:false, replayCheckpoint:"", expansion:{}};
const fields = ["base_model","train_file","validation_file","epochs","batch_size","accumulation_steps","max_length","learning_rate","memory_budget_mb","resume_from","lora_rank","lora_alpha","seed","balance_classes"];
const textFields = new Set(["base_model","train_file","validation_file","resume_from"]);
let toastTimer;
let annotationWrites = Promise.resolve();
function toast(message, error=false) { $("toast").textContent=message; $("toast").classList.toggle("error",error); $("toast").style.display="block"; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$("toast").style.display="none",error?10000:4500); }
async function call(name,...args) { if(!app.api) throw new Error("桌面桥接未连接，请通过 run.bat 打开界面。"); const result=await app.api[name](...args); if(!result.ok) throw new Error(result.error); return result.data; }
function bind(id, handler) { $(id).addEventListener("click", async()=>{ const button=$(id); if(button.disabled)return; button.disabled=true; try { await handler(); } catch(error) { toast(error.message,true); } finally { button.disabled=false; if(id==="stop-training")button.disabled=!app.busy; if(id==="start-training")button.disabled=app.busy; } }); }
function page(name) { document.querySelectorAll(".page").forEach(el=>el.classList.toggle("active",el.id===`${name}-page`)); document.querySelectorAll(".nav-item").forEach(el=>el.classList.toggle("active",el.dataset.page===name)); $("page-breadcrumb").textContent="工作空间 / "+({training:"模型训练",dataset:"逐句标注",analysis:"本地分析"}[name]); if(name==="training")drawCharts(); }
function configFromUI() { const result={...app.config}; for(const key of fields) result[key]=key==="balance_classes"?$(key).checked:textFields.has(key)?$(key).value.trim():Number($(key).value); return result; }
function applyConfig(config) { app.config=config; for(const key of fields)if(config[key]!==undefined){if(key==="balance_classes")$(key).checked=config[key];else $(key).value=config[key];} updateBatch(); }
function updateBatch() { $("effective-batch").textContent=`有效批大小 ${Number($("batch_size").value)*Number($("accumulation_steps").value)} · LoRA rank ${$("lora_rank").value}`; $("sampling-caption").textContent=app.config.sampling_max_ratio?`温和平衡，单句抽样权重最高 ${app.config.sampling_max_ratio} 倍；评估保留原始分布。`:"让各类别获得相近抽样机会；评估保留原始分布。"; }
function percent(value,digits=1){return value==null?"—":(100*value).toFixed(digits)+"%";}
function fixed(value,digits=4){return value==null?"—":Number(value).toFixed(digits);}

function plot(canvas, metric, isAccuracy=false) {
  const width=canvas.clientWidth, height=195; if(!width)return;
  const ratio=window.devicePixelRatio||1; canvas.width=width*ratio;canvas.height=height*ratio;
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
  const box={x:42,y:15,w:width-58,h:142}; ctx.font="10px Segoe UI, Microsoft YaHei";
  const values=app.history.flatMap(row=>[row.train[metric],row.validation[metric]]);
  const max=isAccuracy?1:Math.max(.1,...values)*1.12;
  for(let i=0;i<=4;i++){const y=box.y+box.h*(1-i/4);ctx.strokeStyle="#e9ede5";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(box.x,y);ctx.lineTo(box.x+box.w,y);ctx.stroke();ctx.fillStyle="#a0aaa0";ctx.textAlign="right";ctx.fillText(isAccuracy?`${i*25}%`:(max*i/4).toFixed(2),box.x-9,y+3);}
  if(!app.history.length){ctx.fillStyle="#abb4a9";ctx.textAlign="center";ctx.fillText("训练后显示实际指标",box.x+box.w/2,box.y+box.h/2);return;}
  const maxEpoch=Math.max(2,app.history.at(-1).epoch);
  for(const [key,color] of [["train","#3f8976"],["validation","#6284b7"]]){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;let last;app.history.forEach((row,index)=>{const x=box.x+box.w*(row.epoch-1)/(maxEpoch-1),y=box.y+box.h*(1-row[key][metric]/max);if(index)ctx.lineTo(x,y);else ctx.moveTo(x,y);last=[x,y];});ctx.stroke();if(last){ctx.fillStyle=color;ctx.beginPath();ctx.arc(...last,3,0,2*Math.PI);ctx.fill();}}
  ctx.fillStyle="#9ba79c";ctx.textAlign="left";ctx.fillText("1",box.x,181);ctx.textAlign="right";ctx.fillText(String(app.history.at(-1).epoch),box.x+box.w,181);ctx.textAlign="center";ctx.fillText("训练轮次",box.x+box.w/2,181);
}
function drawCharts(){plot($("loss-chart"),"loss");plot($("accuracy-chart"),"accuracy",true);}
function renderConfusion(record) {
  const box=$("confusion"), matrix=record.validation.confusion_matrix;box.replaceChildren();box.className="matrix-grid";box.style.gridTemplateColumns=`36px repeat(${app.labels.length},1fr)`;
  function cell(value,axis=false,opacity=0){const el=document.createElement("div");el.textContent=value;if(axis)el.className="axis";else el.style.backgroundColor=`rgba(63,137,118,${opacity})`;if(opacity>.55)el.style.color="#fff";box.append(el);}
  cell("",true);app.labels.forEach(label=>cell(label,true));const max=Math.max(1,...matrix.flat());
  matrix.forEach((row,i)=>{cell(app.labels[i],true);row.forEach(value=>cell(value,false,.04+.8*value/max));});
  $("matrix-epoch").textContent=`第 ${record.epoch} 轮`;
  $("label-metrics").replaceChildren();record.validation.per_label.forEach(item=>{const tr=document.createElement("tr");[item.label,percent(item.precision),percent(item.recall),fixed(item.f1,3),item.support].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});$("label-metrics").append(tr);});
}
function renderTraining(state) {
  const running=["starting","loading","training","evaluating"].includes(state.state)&&state.running!==false;
  app.busy=running;document.body.classList.toggle("training-busy",running);$("start-training").disabled=running;$("stop-training").disabled=!running;
  const names={idle:"就绪",starting:"准备中",loading:"加载模型",training:"训练中",evaluating:"评估中",completed:"已完成",cancelled:"已停止",failed:"失败"};
  $("run-state").textContent=names[state.state]||state.state||"就绪";
  const epochs=state.config?.epochs||Number($("epochs").value),epoch=state.epoch||0;
  $("epoch-value").replaceChildren(document.createTextNode(`${epoch} `));const sub=document.createElement("small");sub.textContent=`/ ${epochs} 轮`;$("epoch-value").append(sub);
  const progress=state.total_steps?100*(state.step||0)/state.total_steps:0;$("training-progress").style.width=`${progress}%`;
  if(state.peak_allocated_mb!=null)$("memory-value").textContent=`${Math.round(state.peak_allocated_mb)} MB`;
  if(state.peak_reserved_mb!=null)$("reserved-value").textContent=`峰值保留 ${Math.round(state.peak_reserved_mb)} MB`;
  $("training-message").textContent=state.error||state.message||"等待开始。首次使用请先准备基础模型。";
  if(state.live_train_loss!=null)$("train-loss-value").textContent=`当前批次训练损失 ${fixed(state.live_train_loss)}`;
  if(state.best_epoch)$("training-message").textContent+=` 最佳模型：第 ${state.best_epoch} 轮 · ${state.best_model||""}`;
  if(state.config?.train_file)$("training-message").textContent+=` 本次记录的数据集：${state.config.train_file}`;
  const next=state.history||[];
  if(next.length!==app.history.length||next.at(-1)?.validation.loss!==app.history.at(-1)?.validation.loss){
    app.history=next;
    drawCharts();
    if(next.length){
      const last=next.at(-1);
      $("accuracy-value").textContent=percent(last.validation.accuracy);
      $("loss-value").textContent=fixed(last.validation.loss);
      $("f1-value").textContent=`验证宏 F1 · ${fixed(last.validation.macro_f1,3)}`;
      renderConfusion(last);
    } else {
      $("accuracy-value").textContent="—";
      $("loss-value").textContent="—";
      $("f1-value").textContent="验证宏 F1 · —";
      $("confusion").replaceChildren();
      $("label-metrics").replaceChildren();
      $("matrix-epoch").textContent="等待训练";
    }
  }
}
async function poll(){if(!app.api||app.polling)return;app.polling=true;try{const state=await call("poll");if(!app.replay||state.training.running)renderTraining(state.training);const d=state.download;if(["starting","downloading"].includes(d.state)){$("model-ready").textContent="下载中";$("download-model").disabled=true;}else{$("download-model").disabled=app.busy;if(d.state==="completed")$("model-ready").textContent="已就绪";if(d.state==="failed"){$("model-ready").textContent="下载失败";$("model-ready").title=d.error;}}}catch(error){console.error(error);}finally{app.polling=false;}}

const contentNames={light_novel_dialogue:"轻小说对白",light_novel_narration:"轻小说旁白",visual_novel_dialogue:"Galgame 对白",acg_review:"ACG 评论",acg_introduction:"作品介绍",unknown:"未分类"};
function datasetStats(stats){
  $("train-file-caption").textContent=`训练集 · ${stats.count.toLocaleString()} 句（标注页）`;
  const acg=stats.count>0&&stats.domains?.acg===stats.count;
  $("training-data-note").textContent=acg?"ACG 语料使用独立模型一致性初标，保留来源与复核状态；验证集按整部作品隔离，自动分数未经校准。":stats.path.includes("expanded")?"历史混合扩充集包含公版小说；请使用新版 ACG 训练集。":"训练与验证指标反映所选文件的参考标签；训练前应检查标签质量。";
  app.count=stats.count;$("sample-count").textContent=`${stats.count.toLocaleString()} 句`;
  $("dataset-distribution").textContent=Object.entries(stats.labels).map(([key,value])=>`${key} ${value}`).join(" · ");
  $("dataset-file").textContent=stats.path;
  $("corpus-quality").hidden=!acg;
  if(acg){
    $("corpus-work-count").textContent=`${stats.unique_works} 部作品 / 系列`;
    const container=$("corpus-sources");container.replaceChildren();
    Object.entries(stats.content_types).sort((a,b)=>b[1]-a[1]).forEach(([kind,count])=>{
      const row=document.createElement("div"),caption=document.createElement("span"),track=document.createElement("div"),fill=document.createElement("i"),value=document.createElement("span");
      row.className="corpus-bar";caption.textContent=contentNames[kind]||kind;track.className="corpus-track";fill.style.width=`${100*count/stats.count}%`;value.textContent=`${count.toLocaleString()} · ${percent(count/stats.count,0)}`;track.append(fill);row.append(caption,track,value);container.append(row);
    });
    const review=app.expansion.review||{};
    $("corpus-review").textContent=`当前文件：助手已复核 ${(stats.review_statuses?.assistant_reviewed||0).toLocaleString()} 句，用户已复核 ${(stats.review_statuses?.user_reviewed||0).toLocaleString()} 句。独立校准阅读 ${review.calibration_read||0} 句，盲读测试参考 ${app.expansion.test_reference_count||0} 句。其余为自动初标。`;
  }
}
function queueAnnotation(index, text, label) {
  annotationWrites = annotationWrites.catch(()=>{}).then(async()=>{
    datasetStats(await call("update_sample", index, text, label));
  });
  annotationWrites.catch(error=>toast(error.message,true));
}
async function datasetPage() {
  await annotationWrites;
  const data=await call("dataset_page",app.offset,app.pageSize,$("filter-label").value,$("filter-review").value);
  datasetStats(data.stats);
  app.count=data.total??data.stats.count;
  $("dataset-rows").replaceChildren();
  data.rows.forEach((sample,index)=>{
    const absoluteIndex=data.indices?.[index]??app.offset+index;
    const tr=document.createElement("tr"), number=document.createElement("td"), sentence=document.createElement("td"), label=document.createElement("td"), select=document.createElement("select");
    number.textContent=String(absoluteIndex+1).padStart(4,"0");
    sentence.textContent=sample.text;
    sentence.contentEditable="true";
    select.append(new Option("未标注",""));
    app.labels.forEach(value=>select.append(new Option(value,value)));
    select.value=sample.label;
    const save=()=>queueAnnotation(absoluteIndex,sentence.textContent,select.value);
    sentence.addEventListener("blur",save);
    select.addEventListener("change",save);
    label.append(select);
    const provenance=document.createElement("td");
    provenance.className="sample-provenance";
    const method=({independent_teacher:"自动初标",independent_teacher_consensus:"模型一致性初标",user_edit:"用户修改",assistant_review:"助手复核",assistant_blind_review:"助手盲读参考",authored_synthetic:"合成启动",unlabeled:"未标注"})[sample.label_method]||(sample.source==="authored_synthetic"?"合成启动":"未记录标注方式");
    const detail=document.createElement("details"),summary=document.createElement("summary"),body=document.createElement("div");
    summary.textContent=`${sample.title||sample.source||method}${sample.confidence!=null?" · "+percent(sample.confidence,0):""}`;
    body.className="provenance-details";
    const info=[`${method} / ${sample.review_status||"not_reviewed"}`,sample.teacher_label?`原始教师标签：${sample.teacher_label}；当前助手参考标签：${sample.label}`:"",sample.source_text_edited?`用户已修改目标文本，原始出处对应修改前的句子：${sample.original_source_text||"见上下文"}`:"",sample.supporting_teachers?`支持标签的模型：${sample.supporting_teachers.join("、")}`:"",sample.context?`原文上下文（仅供复核）：${sample.context}`:"",sample.review_reason?`复核理由：${sample.review_reason}`:"",sample.source_url||"",sample.location||""];
    if(sample.scores&&typeof Object.values(sample.scores)[0]==="object")Object.entries(sample.scores).forEach(([name,scores])=>info.push(`${name}：${Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0,2).map(([label,score])=>`${label} ${percent(score,0)}`).join(" / ")}`));
    info.filter(Boolean).forEach(value=>{const p=document.createElement("p");p.textContent=value;body.append(p);});
    const note=document.createElement("p");note.textContent="初标分数不是校准后的准确率。";body.append(note);detail.append(summary,body);provenance.append(detail);
    tr.append(number,sentence,label,provenance);
    $("dataset-rows").append(tr);
  });
  $("page-number").textContent=`${Math.floor(app.offset/app.pageSize)+1} / ${Math.max(1,Math.ceil(app.count/app.pageSize))}`;
  $("previous-page").disabled=app.offset===0;
  $("next-page").disabled=app.offset+app.pageSize>=app.count;
}
async function refreshRuns(){app.runs=await call("list_runs");const current=$("checkpoint").value;$("checkpoint").replaceChildren(new Option("选择已训练的模型…",""));app.runs.forEach(run=>$("checkpoint").append(new Option(`${run.name} · 第 ${run.epoch} 轮 · F1 ${fixed(run.f1,3)}`,run.path)));if(app.runs.some(run=>run.path===current))$("checkpoint").value=current;else if(app.runs.length)$("checkpoint").value=app.runs[0].path;}
function showPredictions(results){const container=$("analysis-results");container.className="";container.replaceChildren();results.forEach((prediction,index)=>{const card=document.createElement("article");card.className="prediction";const head=document.createElement("div");head.className="prediction-head";const number=document.createElement("span"),text=document.createElement("p"),label=document.createElement("span");number.className="prediction-index";number.textContent=String(index+1).padStart(2,"0");text.className="prediction-text";text.textContent=prediction.text;label.className="prediction-label";label.textContent=`${prediction.label} · ${percent(prediction.confidence)}`;head.append(number,text,label);card.append(head);const bars=document.createElement("div");bars.className="probability-bars";Object.entries(prediction.probabilities).forEach(([name,value])=>{const item=document.createElement("div"),caption=document.createElement("span"),track=document.createElement("div"),fill=document.createElement("div"),score=document.createElement("span");item.className="probability";caption.textContent=name;track.className="bar-track";fill.className="bar-fill";fill.style.width=`${value*100}%`;score.className="bar-value";score.textContent=percent(value);track.append(fill);item.append(caption,track,score);bars.append(item);});card.append(bars);if(prediction.truncated){const note=document.createElement("p");note.className="muted";note.textContent="该句超过模型序列长度，已截断；可拆成更短的句子重试。";card.append(note);}container.append(card);});}

document.querySelectorAll(".nav-item").forEach(button=>button.addEventListener("click",()=>page(button.dataset.page)));
for(const key of ["batch_size","accumulation_steps","lora_rank"])$(key).addEventListener("input",updateBatch);
bind("start-training",async()=>{await call("start_training",configFromUI());app.busy=true;app.replay=false;app.replayCheckpoint="";toast("训练进程已启动；分析模型已卸载。");await poll();});
bind("stop-training",async()=>{await call("stop_training");toast("已请求停止，当前批次结束后保存已有最佳模型。");});
bind("download-model",async()=>{await call("download_model");toast("基础模型下载已开始，可继续查看或标注数据。");});
bind("export-metrics",async()=>{const path=await call("export_metrics",app.replay?app.replayCheckpoint:null);if(path)toast("指标已导出。");});
bind("previous-page",async()=>{app.offset=Math.max(0,app.offset-app.pageSize);await datasetPage();});
bind("next-page",async()=>{if(app.offset+app.pageSize<app.count)app.offset+=app.pageSize;await datasetPage();});
bind("add-samples",async()=>{datasetStats(await call("add_samples",$("new-samples").value,$("new-label").value));$("new-samples").value="";app.offset=Math.floor((app.count-1)/app.pageSize)*app.pageSize;await datasetPage();});
bind("import-dataset",async()=>{await annotationWrites;const stats=await call("import_dataset");if(stats){app.offset=0;await datasetPage();}});
bind("save-dataset",async()=>{await annotationWrites;const stats=await call("save_dataset");if(stats){datasetStats(stats);$("train_file").value=stats.path;toast("标注已保存，训练集路径已更新。");}});
bind("use-expanded",async()=>{await annotationWrites;const result=await call("use_expanded_dataset");app.offset=0;$("filter-label").value="";$("filter-review").value="";$("train_file").value=result.preset.train_file;$("validation_file").value=result.preset.validation_file;$("max_length").value="256";$("balance_classes").checked=true;app.config.sampling_power=.5;app.config.sampling_max_ratio=10;updateBatch();await datasetPage();toast("已选择 ACG 训练集并启用均衡抽样，抽样权重已设上限。来源、上下文和分数可逐句查看。");});
for(const id of ["filter-label","filter-review"])$(id).addEventListener("change",async()=>{try{app.offset=0;await datasetPage();}catch(error){toast(error.message,true);}});
bind("refresh-runs",refreshRuns);
bind("load-analysis",async()=>{const checkpoint=$("checkpoint").value;if(!checkpoint)throw new Error("请先完成训练或选择已有模型。");const result=await call("load_analysis",checkpoint,$("analysis-device").value);$("analysis-status").textContent=`已连接 · ${result.device.toUpperCase()} · 首次分析加载权重`;toast("独立分析服务已就绪。");});
bind("unload-analysis",async()=>{await call("unload_analysis");$("analysis-status").textContent="分析进程已退出，显存已释放。";});
bind("predict",async()=>{const result=await call("predict",$("analysis-text").value);showPredictions(result.predictions);$("analysis-status").textContent=`模型已加载 · ${result.model.device.toUpperCase()} · ${result.predictions.length} 句`;});
bind("view-run-metrics",async()=>{const status=await call("load_run_metrics",$("checkpoint").value);app.history=[];app.replay=true;app.replayCheckpoint=$("checkpoint").value;renderTraining({...status,running:false});page("training");});
bind("resume-selected",async()=>{if(!$("checkpoint").value)throw new Error("请选择模型。");const status=await call("load_run_metrics",$("checkpoint").value);applyConfig({...status.config,train_file:$("train_file").value,validation_file:$("validation_file").value,resume_from:$("checkpoint").value});page("training");toast("已恢复模型设置并保留当前训练集路径；继续训练会创建新的训练目录。");});
window.addEventListener("resize",drawCharts);
window.addEventListener("pywebviewready",async()=>{
  app.api=window.pywebview.api;
  try{
    const data=await call("bootstrap");app.labels=data.labels;app.expansion=data.expansion||{};applyConfig(data.config);
    $("use-expanded").disabled=!app.expansion.ready;
    if(app.expansion.ready)$("dataset-notice").textContent=`ACG 训练集 ${app.expansion.train_count.toLocaleString()} 句：轻小说、Galgame 对白与 ACG 评论；独立模型一致性初标，助手抽样复核。自动分数不等于标注准确率。`;
    $("connection").textContent="本地桌面已连接";$("connection").classList.add("ready");
    $("model-ready").textContent=data.base_ready?"已就绪":"尚未准备";$("service-file").textContent=data.service_file;
    app.labels.forEach(label=>{$("new-label").append(new Option(label,label));$("filter-label").append(new Option(label,label));});
    await datasetPage();await refreshRuns();
    if(app.runs.length){
      const checkpoint=$("checkpoint").value,status=await call("load_run_metrics",checkpoint);
      if(status.config?.train_file===data.config.train_file&&status.config?.validation_file===data.config.validation_file){
        app.replay=true;app.replayCheckpoint=checkpoint;renderTraining({...status,running:false});
      }
    }
    call("hardware").then(data=>{const parts=data.gpu.split(",").map(p=>p.trim());$("gpu-name").textContent=parts[0];$("gpu-memory").textContent=`${parts[1]} MB 总显存 · 当前 ${parts[2]} MB`;}).catch(error=>{$("gpu-name").textContent="CUDA 待配置";$("gpu-memory").textContent=error.message;});
    await poll();setInterval(poll,1200);
  }catch(error){toast(error.message,true);}
});
drawCharts();
