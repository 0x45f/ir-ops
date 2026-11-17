"""Compare LLVM inline PTX cache hints and warp counts, including cold-cache timing."""
from pathlib import Path
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
    out=Path('perf_cache_tuning');out.mkdir(exist_ok=True)
    torch.manual_seed(0);results=[]
    for cols in [2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda'); y=torch.empty_like(x); expected=torch.softmax(x,1)
        for threads in [128,256]:
            block=(cols+threads*4-1)//(threads*4)*(threads*4)
            p=write_ir(block,threads,False,out/f't{threads}')
            original=p.read_text()
            for load,store in [('', ''),('.cg',''),('.cs',''),('','.cs'),('.cg','.cs'),('','.wt')]:
                ir=original.replace('ld.global.v4',f'ld.global{load}.v4').replace('st.global.v4',f'st.global{store}.v4')
                exe=_compile_llvm_ir(0,ir,threads//32,threads//8,1,False)
                fn=lambda: _launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
                fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
                ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
                row=dict(cols=cols,threads=threads,load=load,store=store,us=ms*1000,registers=exe.n_regs)
                results.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
