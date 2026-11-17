"""DIAGNOSTIC ONLY: paired zero-compute copy and real softmax across all shapes.

Copy timings are not a softmax implementation, nor a rigorous hardware bound.
"""
import hashlib,json,math,statistics
from functools import lru_cache
from pathlib import Path
import torch,triton
import triton_softmax as reference
from softmax_llir import _compile_llvm_ir,_launch,_select_block,softmax_from_llir
from edit_softmax_ir import write_ir

OUT=Path('perf_memory_floor');OUT.mkdir(exist_ok=True)
@lru_cache(maxsize=64)
def copy_executable(cols):
    block=_select_block(cols)
    path=write_ir(block,128,directory=OUT/'copy_ir',tail_only=True,copy_only=True)
    warp=block<=1024
    return _compile_llvm_ir(0,path.read_text(),4,0 if warp else 16,4 if warp else 1,False)

def copy_fn(x):
    y=torch.empty_like(x)
    exe=copy_executable(x.shape[1])
    _launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
    return y

def main():
    torch.manual_seed(0);rows=[]
    metadata=dict(kind='DIAGNOSTIC ONLY; copy is not softmax or a proven lower bound',rounds=3,warmup_ms=20,rep_ms=60,
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
                  reference_sha256=hashlib.sha256(Path('triton_softmax.py').read_bytes()).hexdigest())
    for cols in range(256,12800,128):
        x=torch.randn(4096,cols,device='cuda')
        torch.testing.assert_close(copy_fn(x),x,rtol=0,atol=0)
        expected=torch.softmax(x,1)
        torch.testing.assert_close(reference.softmax(x),expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(softmax_from_llir(x),expected,rtol=1e-5,atol=1e-6)
        functions={'reference':lambda:reference.softmax(x),'softmax':lambda:softmax_from_llir(x),'copy':lambda:copy_fn(x)}
        times={k:[] for k in functions}
        names=list(functions)
        for index in range(3):
            order=names[index:]+names[:index]
            for name in order:
                times[name].append(1000*triton.testing.do_bench(functions[name],warmup=20,rep=60,return_mode='median'))
        medians={k:statistics.median(v) for k,v in times.items()}
        row=dict(cols=cols,rounds=times,median_us=medians,softmax_speedup=medians['reference']/medians['softmax'],
                 copy_comparator_ratio=medians['reference']/medians['copy'])
        rows.append(row)
        print(f'N={cols}: ref={medians["reference"]:.3f}, softmax={medians["softmax"]:.3f}, COPY={medians["copy"]:.3f} us',flush=True)
        (OUT/'results.json').write_text(json.dumps(dict(metadata=metadata,results=rows),indent=2))
    summary=dict(kind='DIAGNOSTIC ONLY',shapes=len(rows),
                 actual_softmax_geomean=math.exp(statistics.mean(math.log(r['softmax_speedup']) for r in rows)),
                 copy_comparator_geomean=math.exp(statistics.mean(math.log(r['copy_comparator_ratio']) for r in rows)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
