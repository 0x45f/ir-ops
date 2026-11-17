"""Explore direct-IR persistent row loops against the previous best kernel."""
import json
from pathlib import Path
from dataclasses import replace
import torch
import triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir, _get_executable, _launch, _select_block


def main():
    torch.manual_seed(0)
    results=[]
    out=Path('perf_persistent_tuning'); out.mkdir(exist_ok=True)
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    for cols in [1024,2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda'); y=torch.empty_like(x)
        expected=torch.softmax(x,1)
        block=_select_block(cols)
        base=_get_executable(0,block)
        variants=[('previous',base)]
        for threads in [128,256]:
            warp=cols<=1024
            unit=128 if warp else threads*4
            capacity=(cols+unit-1)//unit*unit
            p=write_ir(capacity,threads,warp,out/f't{threads}',persistent=True)
            exe=_compile_llvm_ir(0,p.read_text(),threads//32,0 if warp else threads//8,threads//32 if warp else 1,True)
            for blocks_per_sm in [1,2,4,8,16]:
                variants.append((f't{threads}_g{blocks_per_sm}',replace(exe,num_programs=sm*blocks_per_sm)))
        for name,exe in variants:
            fn=lambda: _launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
            fn(); torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
            ms=triton.testing.do_bench_cudagraph(fn,rep=15,return_mode='median')
            row=dict(cols=cols,name=name,us=ms*1000,registers=exe.n_regs,grid=exe.num_programs)
            results.append(row); print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
