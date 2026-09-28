"""Recompute timing summaries using the standard library; no model imports."""
import argparse
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CPUS = ('12700H', '13900HX', '14900HX')

def validate(runs):
    expected = {f'{c}-{p}-{d}' for c in CPUS for p in ('image','h3') for d in ('cpu','cuda')}
    expected |= {f'{c}-h3-cuda-reserve3' for c in CPUS[:2]}
    if set(runs) != expected:
        raise ValueError('Expected exactly 12 main groups and 2 headroom groups')
    for name, r in runs.items():
        if r['status'] != 'success':
            raise ValueError(name + ': unsuccessful run')
        cases = ('image_short','image_long') if r['arguments']['profile']=='image' else ('h3_first_frame',)
        for case in cases:
            samples = [s for s in r['samples'] if s['case']==case]
            if [s['repeat'] for s in samples] != [1,2,3,4]:
                raise ValueError(name + ': missing or duplicate repeat')
            if any(not s['finite'] or s['encode_s'] <= 0 for s in samples):
                raise ValueError(name + ': invalid output or timing')
        if r['arguments']['device']=='cpu' and (r['cuda_initialized_final'] or r['placement_after']['devices'] != ['cpu']):
            raise ValueError(name + ': CPU isolation failed')
    if sum(len(r['samples']) for r in runs.values()) != 80:
        raise ValueError('Expected 80 measured forwards')

def values(runs, name, case):
    return [s['encode_s'] for s in runs[name]['samples'] if s['case']==case]

def render(runs):
    validate(runs)
    lines=['# 全部正式计时记录','',
        '单位秒。每个用例4次真实编码；后3次取中位数并非长期稳态保证。GPU默认组与显存预留组分别保留。', '',
        '| 组别 | 用例 | 首次 | 第2次 | 第3次 | 第4次 | 后3次中位数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        for case in dict.fromkeys(s['case'] for s in r['samples']):
            vs=values(runs,name,case)
            lines.append('| '+name+' | '+case+' | '+' | '.join(f'{v:.3f}' for v in [*vs,statistics.median(vs[1:])])+' |')
    lines += ['', '## CPU内存采样', '',
        '单位GiB；工作集与私有提交不可相加。内存每0.5秒采样，可能漏掉短峰值。系统可用量含当时后台占用。','',
        '| 组别 | 峰值工作集 | 峰值私有提交 | 最低整机可用RAM |','|---|---:|---:|---:|']
    for name,r in runs.items():
        if r['arguments']['device']=='cpu':
            m=r['memory']['overall']
            lines.append('| '+name+' | '+' | '.join(f'{m[k]/1024**3:.2f}' for k in ('rss_peak_bytes','private_peak_bytes','system_available_min_bytes'))+' |')
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BASE/'results/2026-09-28/runs.json')
    parser.add_argument('--write',type=Path,help='Optionally write the report instead of printing it')
    args=parser.parse_args()
    runs=json.loads(args.input.read_text(encoding='utf-8'))
    report=render(runs)
    if args.write:
        args.write.write_text(report,encoding='utf-8',newline='\n')
    else:
        print(report)

if __name__=='__main__':
    main()
