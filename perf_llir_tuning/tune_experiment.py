import sys
sys.path.insert(0, '/data/wangzhen/ir-ops')
import json, math
from pathlib import Path
import torch, triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir, _launch
results=[]
for cols in [1024, 2048, 4096, 5504, 8192, 8320, 12672]:
    x=torch.randn(4096,cols,device='cuda'); y=torch.empty_like(x)
    expected=torch.softmax(x,1)
    for threads,warp in [(128,True),(128,False),(256,False),(512,False)]:
        if warp and cols>4096: continue
        for exact in [False,True]:
            unit=(32 if warp else threads)*4
            block=math.ceil(cols/unit)*unit if exact else max(unit,triton.next_power_of_2(cols))
            directory=f'/tmp/ir_ops_candidates/t{threads}_w{int(warp)}'
            path=write_ir(block,threads,warp,directory)
            exe=_compile_llvm_ir(0,path.read_text(),threads//32,0 if warp else threads//8,threads//32 if warp else 1,False)
            fn=lambda: _launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
            fn()
            torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
            ms=triton.testing.do_bench_cudagraph(fn,rep=20,return_mode='median')
            r=dict(cols=cols,threads=threads,warp=warp,block=block,registers=exe.n_regs,us=ms*1000,path=str(path))
            results.append(r); print(json.dumps(r),flush=True)
Path('/tmp/ir_ops_tune.json').write_text(json.dumps(results,indent=2))
