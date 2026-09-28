"""Render publication figures from archived data; never runs a model."""
import json, os, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'assets'; OUT.mkdir(exist_ok=True)
candidates=[Path(os.environ.get('WINDIR',''))/'Fonts'/'msyh.ttc']
candidates += [Path(p) for p in font_manager.findSystemFonts() if any(s in Path(p).name.lower() for s in ('notosanscjk','sourcehansans','msyh'))]
font=next((p for p in candidates if p.is_file()),None)
if font is None: raise SystemExit('Install a Chinese font (Microsoft YaHei or Noto Sans CJK) before rendering figures.')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':13,'savefig.facecolor':'white','svg.fonttype':'path'})
runs=json.loads((BASE/'results/2026-09-28/runs.json').read_text(encoding='utf-8'))
ink='#18344B'; blue='#2878B8'; teal='#008B80'; red='#C15A45'; gray='#607386'
def vals(c,suffix='cpu'):
    return [s['encode_s'] for s in runs[f'{c}-h3-{suffix}']['samples']]
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    svg=OUT/(name+'.svg')
    svg.write_text(svg.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(11,7.2)); fig.subplots_adjust(left=.30,right=.95,top=.80,bottom=.23)
fig.text(.05,.95,'H3 文本编码：同一台机器，CPU 与 GPU 对照',size=20,weight='bold',color=ink)
fig.text(.05,.89,'固定首帧输入 · 后3次中位数 · 单位：秒',size=13,color=gray)
cs=['12700H','13900HX','14900HX']; gs=['3080 Ti Laptop 16GB','5080 Laptop 16GB','5090 Laptop 24GB']
for i,(c,g) in enumerate(zip(cs,gs)):
    y=4-2*i; cv=statistics.median(vals(c)[1:]); gv=statistics.median(vals(c,'cuda-reserve3' if i<2 else 'cuda')[1:])
    ax.barh(y+.23,cv,height=.38,color=blue,label='纯CPU' if i==0 else None)
    ax.barh(y-.23,gv,height=.38,color=teal,label='GPU路径' if i==0 else None)
    ax.text(cv+2,y+.23,f'{cv:.2f}',va='center',size=14,color=blue,weight='bold')
    ax.text(gv+2,y-.23,f'{gv:.3f}',va='center',size=14,color=teal,weight='bold')
ax.set_yticks([4,2,0],[f'{c}\nRTX {g}' for c,g in zip(cs,gs)],color=ink)
ax.set_xlim(0,150); ax.set_xlabel('编码时间 / 秒',color=gray); ax.set_axisbelow(True); ax.grid(axis='x',alpha=.17)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(axis='y',length=0,pad=12); fig.legend(loc='upper right',bbox_to_anchor=(.95,.865),frameon=False,ncol=2)
fig.text(.05,.12,'16GB卡：预留3GiB显存，GPU计算＋CPU卸载；5090：默认完整驻留。',size=12,color=ink)
fig.text(.05,.075,'只测TE，不含视频采样或解码。H3 CPU/GPU相对RMSE约5.2%，画质等价未验收。',size=11,color=gray)
fig.text(.05,.035,'2026-09-28 · cpu-te-bench · 来源：仓库 runs.json / comparisons.json',size=10,color=gray)
save(fig,'h3-cpu-gpu')

fig,axs=plt.subplots(1,2,figsize=(11,6.4)); fig.subplots_adjust(left=.08,right=.96,top=.75,bottom=.26,wspace=.27)
fig.text(.05,.95,'显存预留前后：四次编码全部保留',size=21,weight='bold',color=ink)
fig.text(.05,.89,'相同权重与输入，只改变加载器 reserve-vram=3GiB 设置',size=13,color=gray)
for ax,c,g in zip(axs,cs[:2],gs[:2]):
    v=vals(c,'cuda'); w=vals(c,'cuda-reserve3')
    ax.plot(range(1,5),v,'o-',color=red,lw=2.2,label='默认配置')
    ax.plot(range(1,5),w,'o-',color=teal,lw=2.2,label='预留3GiB')
    for x,y in enumerate(v,1): ax.annotate(f'{y:.1f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',size=11,color=red)
    for x,y in enumerate(w,1): ax.annotate(f'{y:.2f}',(x,y),xytext=(0,-17),textcoords='offset points',ha='center',size=10,color=teal)
    ax.set_title(f'{c} / RTX {g}',size=12,color=ink,pad=16)
    ax.set_xticks(range(1,5),['首次','第2次','第3次','第4次']); ax.set_ylim(-max(v)*.13,max(v)*1.22)
    ax.set_ylabel('编码时间 / 秒'); ax.grid(alpha=.15)
    for sp in ax.spines.values():sp.set_visible(False)
axs[0].legend(loc='upper left',frameon=False,fontsize=11)
fig.text(.05,.14,'两台机器调整前后，GPU输出分别逐元素相同。预留后仍由GPU完成主要计算。',size=12,color=ink)
fig.text(.05,.085,'左右纵轴范围不同；3GiB未经过全范围寻优。后续三次成绩不等于长期吞吐。',size=11,color=gray)
fig.text(.05,.035,'2026-09-28 · cpu-te-bench · 来源：仓库全部4次正式计时',size=10,color=gray)
save(fig,'vram-headroom')

fig,ax=plt.subplots(figsize=(11,6.5)); ax.set_xlim(0,11); ax.set_ylim(0,6.5); ax.axis('off')
ax.text(.2,6.1,'权重放哪里，与计算在哪里，是两个问题',fontsize=21,color=ink,weight='bold')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.12',ec=color,fc=color+'12',lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13,color=ink)
def arrow(x1,y1,x2,y2,label=''):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,color=gray,lw=1.5))
    if label:ax.text((x1+x2)/2,y1+.18,label,ha='center',fontsize=10,color=gray)
ax.text(.2,5.2,'纯CPU TE',fontsize=16,color=blue,weight='bold')
box(.3,3.9,2.7,.85,'系统内存\n完整模型权重',blue)
box(4.0,3.9,2.7,.85,'CPU\n反量化与前向计算',blue)
box(8.0,3.9,2.3,.85,'条件张量\n交给后续任务',blue)
arrow(3.18,4.32,3.82,4.32); arrow(6.85,4.32,7.82,4.32)
ax.text(.2,3.05,'GPU计算＋CPU卸载',fontsize=16,color=teal,weight='bold')
box(.3,1.65,2.7,.95,'系统内存\n暂未常驻GPU的权重',teal)
box(4.0,1.65,2.7,.95,'GPU\n常驻权重＋当前层计算',teal)
box(8.0,1.65,2.3,.95,'条件张量\n交给后续任务',teal)
arrow(3.18,2.12,3.82,2.12,'按需搬运'); arrow(6.85,2.12,7.82,2.12)
ax.text(.3,.85,'显存预留：减少同时常驻的权重，给反量化、激活和临时缓冲腾空间。',fontsize=13,color=ink)
ax.text(.3,.35,'示意图：不删除权重、不跳过模型层；传输与调度成本仍然存在。',fontsize=11,color=gray)
save(fig,'cpu-vs-offload')
print('Rendered 3 PNG and 3 SVG figures from recorded results.')
