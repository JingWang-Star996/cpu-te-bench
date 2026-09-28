"""Isolated paired TE benchmark. Never starts a service or generates media."""
import argparse, collections, hashlib, json, logging, os, platform, re
import subprocess, sys, threading, time, traceback
from pathlib import Path

p = argparse.ArgumentParser()
for key in ('root', 'checkpoint', 'output', 'reference'):
    p.add_argument('--' + key, required=True)
p.add_argument('--profile', choices=['image', 'h3'], required=True)
p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--threads', type=int, required=True)
p.add_argument('--diagnose', action='store_true')
a = p.parse_args()
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
if a.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
started = time.perf_counter(); stage = 'imports'; stop = threading.Event()
result = dict(status='running', host=platform.node(), pid=os.getpid(), arguments=vars(a), samples=[])
(out/'pid.txt').write_text(str(os.getpid()))
events = (out/'events.jsonl').open('w', encoding='utf-8', buffering=1)
memlog = (out/'memory.jsonl').open('w', encoding='utf-8', buffering=1)
gpulog = (out/'gpu.jsonl').open('w', encoding='utf-8', buffering=1)
peaks = collections.defaultdict(lambda: dict(rss_peak_bytes=0, private_peak_bytes=0, system_available_min_bytes=2**63))

def emit(event, **values):
    row = dict(event=event, stage=stage, elapsed_s=round(time.perf_counter()-started, 3), **values)
    line=json.dumps(row, ensure_ascii=False); events.write(line+'\n'); print(line, flush=True)

def save():
    (out/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda:f.read(16*1024**2), b''): h.update(data)
    return h.hexdigest()

import psutil
proc=psutil.Process(); proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
result['priority']=int(proc.nice()); save()

def monitor_memory():
    low=0
    while not stop.is_set():
        try:
            m=proc.memory_info(); vm=psutil.virtual_memory()
            row=dict(stage=stage, elapsed_s=round(time.perf_counter()-started, 3), rss_bytes=m.rss,
                     private_bytes=getattr(m,'private',m.vms), available_bytes=vm.available,
                     page_faults=getattr(m,'num_page_faults',None), cpu_per_core=psutil.cpu_percent(percpu=True))
            memlog.write(json.dumps(row)+'\n')
            for k in ('overall', stage):
                peaks[k]['rss_peak_bytes']=max(peaks[k]['rss_peak_bytes'],m.rss)
                peaks[k]['private_peak_bytes']=max(peaks[k]['private_peak_bytes'],row['private_bytes'])
                peaks[k]['system_available_min_bytes']=min(peaks[k]['system_available_min_bytes'],vm.available)
            low=low+1 if vm.available<3*1024**3 else 0
            if low>=4 or time.perf_counter()-started>2100:
                result.update(status='resource_guard_abort', memory=dict(peaks)); save()
                emit('resource_guard_abort',available_bytes=vm.available); os._exit(3)
        except Exception as e: emit('monitor_error',error=str(e))
        stop.wait(.5)

def monitor_gpu():
    fields='timestamp,name,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem'
    while not stop.is_set():
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=0x08000000)
            gpulog.write(json.dumps(dict(stage=stage,elapsed_s=round(time.perf_counter()-started,3),fields=fields,csv=r.stdout.strip(),rc=r.returncode))+'\n')
        except Exception as e: gpulog.write(json.dumps(dict(error=str(e)))+'\n')
        stop.wait(1)

mon=threading.Thread(target=monitor_memory,daemon=True); mon.start()
gmon=threading.Thread(target=monitor_gpu,daemon=True); gmon.start()
try:
    import torch, numpy as np
    from PIL import Image, ImageOps
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    if a.device=='cpu':
        def reject_cuda(*args,**kwargs): raise RuntimeError('CPU benchmark rejected CUDA initialization')
        torch.cuda._lazy_init=reject_cuda
    sys.path.insert(0,a.root); os.chdir(a.root)
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv=[sys.argv[0]]+(['--cpu','--fp32-text-enc','--disable-dynamic-vram'] if a.device=='cpu' else ['--disable-dynamic-vram'])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    import comfy.sd, comfy.utils, comfy.model_management as mm
    import comfy.quant_ops as qo
    assert mm.get_torch_device().type==a.device
    def sync():
        if a.device=='cuda': torch.cuda.synchronize()
    result['import_s']=time.perf_counter()-started
    source_names=['comfy/sd.py','comfy/ops.py','comfy/model_management.py','comfy/quant_ops.py','comfy/text_encoders/minimax.py','comfy/text_encoders/qwen_image21.py']
    result['environment']=dict(torch=torch.__version__,python=sys.version,threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),ram_total_bytes=psutil.virtual_memory().total,
        source_sha={n:sha(Path(a.root)/n) for n in source_names},torch_config=torch.__config__.show(),
        kernel_backends=qo.ck.list_backends() if getattr(qo,'_CK_AVAILABLE',False) else None)
    if a.device=='cuda':
        result['environment']['gpu']=dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), free_total_bytes=torch.cuda.mem_get_info())
    stage='checkpoint_hash'; t=time.perf_counter()
    result['checkpoint']=dict(bytes=Path(a.checkpoint).stat().st_size,sha256=sha(a.checkpoint))
    expected='8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f' if a.profile=='image' else '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'
    assert result['checkpoint']['sha256']==expected
    result['hash_s']=time.perf_counter()-t; emit('checkpoint_ready',**result['checkpoint'])
    stage='load'; t=time.perf_counter()
    options={'load_device':torch.device('cpu'),'offload_device':torch.device('cpu'),'initial_device':torch.device('cpu'),'dtype':torch.float32} if a.device=='cpu' else {}
    with torch.inference_mode():
        clip=comfy.sd.load_clip(ckpt_paths=[a.checkpoint],embedding_directory=None,
            clip_type=comfy.sd.CLIPType.MINIMAX if a.profile=='h3' else comfy.sd.CLIPType.QWEN_IMAGE,
            model_options=options,disable_dynamic=True)
    sync(); result['load_api_s']=time.perf_counter()-t
    model=clip.cond_stage_model
    def placement():
        params=list(model.parameters())+list(model.buffers())
        return dict(patcher_class=type(clip.patcher).__name__,devices=sorted({str(x.device) for x in params}),types=dict(collections.Counter(type(x).__name__+':'+str(x.dtype) for x in params)),
            load_device=str(clip.patcher.load_device),offload_device=str(clip.patcher.offload_device),
            model_bytes=clip.patcher.model_size(),loaded_bytes=clip.patcher.loaded_size(),
            lowvram=bool(getattr(model,'model_lowvram',False)),offload_buffer_bytes=getattr(model,'model_offload_buffer_memory',0),
            linear_layouts=dict(collections.Counter(str(getattr(m,'layout_type',None))+':full_precision_mm='+str(getattr(m,'_full_precision_mm',None)) for m in model.modules() if hasattr(m,'weight'))))
    result['placement_loaded']=placement(); emit('loaded',load_api_s=result['load_api_s']); save()
    if a.profile=='image':
        cases=[('image_short','A red ceramic teapot on a wooden table, soft daylight, studio photograph.',[]),
               ('image_long','An old library with arched windows, rows of books, brass lamps, carved wooden desks, '*53,[])]
    else:
        stage='prepare_reference'; t=time.perf_counter()
        with Image.open(a.reference) as im:
            picture=torch.from_numpy(np.array(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.float32)/255.).unsqueeze(0)
        picture=comfy.utils.common_upscale(picture.movedim(-1,1),864,480,'lanczos','disabled').movedim(1,-1)
        result['reference']=dict(file_sha256=sha(a.reference),prepared_sha256=hashlib.sha256(picture.numpy().tobytes()).hexdigest(),prepare_s=time.perf_counter()-t)
        cases=[('h3_first_frame','The woman slowly lifts her gaze as rain falls on the neon-lit street. Subtle natural motion, locked camera, one continuous shot.',[picture])]
    def tokenize(prompt,images):
        return clip.tokenize(prompt,images=images) if a.profile=='h3' else clip.tokenize(prompt,images=images,keep_vision=True,prevent_empty_text=True)
    for name,prompt,images in cases:
        previous=None
        for repeat in range(1,5):
            stage=name+'_'+str(repeat); sync()
            if a.device=='cuda': torch.cuda.reset_peak_memory_stats()
            c0=proc.cpu_times(); t=time.perf_counter()
            with torch.inference_mode():
                tokens=tokenize(prompt,images); sync(); tt=time.perf_counter()
                conditioning=clip.encode_from_tokens_scheduled(tokens); sync(); ft=time.perf_counter()
            hidden,extra=conditioning[0]; original_device=str(hidden.device); original_dtype=str(hidden.dtype)
            hidden=hidden.detach().float().cpu(); sync(); copied=time.perf_counter()
            assert torch.isfinite(hidden).all().item()
            if a.device=='cpu':
                assert not torch.cuda.is_initialized()
                assert placement()['devices']==['cpu']
            c1=proc.cpu_times()
            sample=dict(case=name,repeat=repeat,tokenize_s=tt-t,forward_s=ft-tt,encode_s=ft-t,output_copy_s=copied-ft,
                shape=list(hidden.shape),output_dtype=original_dtype,output_device=original_device,finite=True,
                cpu_time_s=c1.user+c1.system-c0.user-c0.system,sha256=hashlib.sha256(hidden.contiguous().numpy().tobytes()).hexdigest())
            if a.device=='cuda': sample['cuda']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if previous is not None: sample['repeat_max_abs_diff']=(hidden-previous).abs().max().item()
            if repeat==1:
                previous=hidden.clone(); np.save(out/(name+'.npy'),hidden.numpy())
            if a.profile=='h3':
                tags=extra['minimax_token_tags'].cpu()
                assert list(hidden.shape)==[1,441,5120] and tags.shape==(441,) and set(tags.tolist())<={0,1}
                sample['token_tags']=dict(counts=dict(collections.Counter(tags.tolist())),sha256=hashlib.sha256(tags.numpy().tobytes()).hexdigest())
            result['samples'].append(sample); result['placement_after']=placement(); emit('forward_complete',**sample); save()
    if a.diagnose and a.device=='cuda':
        stage='diagnostic_profiler'; emit('diagnostic_start')
        try:
            with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                clip.encode_from_tokens_scheduled(tokens); sync()
            rows=[]
            for e in prof.key_averages():
                rows.append(dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                    cpu_us=e.cpu_time_total,self_device_us=getattr(e,'self_device_time_total',0),device_us=getattr(e,'device_time_total',0)))
            result['diagnostic']=dict(status='success',top_cpu=sorted(rows,key=lambda x:x['self_cpu_us'],reverse=True)[:60],
                top_device=sorted(rows,key=lambda x:x['self_device_us'],reverse=True)[:60])
            (out/'profiler-summary.json').write_text(json.dumps(result['diagnostic'],indent=2),encoding='utf-8')
        except Exception as e:
            result['diagnostic']=dict(status='unavailable',error=repr(e)); emit('diagnostic_error',error=repr(e))
    result['status']='success'
except BaseException as exc:
    result.update(status='failed',error=repr(exc),traceback=traceback.format_exc()); traceback.print_exc(); emit('failed',error=repr(exc))
finally:
    stop.set(); mon.join(3); gmon.join(6)
    result.update(elapsed_s=time.perf_counter()-started,memory=dict(peaks))
    if 'torch' in globals(): result['cuda_initialized_final']=torch.cuda.is_initialized()
    save(); emit('finished',status=result['status']); events.close(); memlog.close(); gpulog.close()
sys.exit(0 if result['status']=='success' else 1)
