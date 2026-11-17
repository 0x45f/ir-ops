from pathlib import Path
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
    out=Path('perf_tail_tuning');out.mkdir(exist_ok=True)
    torch.manual_seed(0);results=[]
    for cols in [1024,2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
        for threads in [128,256,512]:
            rt=32 if cols<=1024 else threads
            block=(cols+rt*4-1)//(rt*4)*(rt*4)
            for tail in [False,True]:
                p=write_ir(block,threads,directory=out/f't{threads}_tail{int(tail)}',row_threads=rt,tail_only=tail)
                exe=_compile_llvm_ir(0,p.read_text(),threads//32,0 if rt==32 else threads//8,threads//rt,False)
                fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
                fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
                ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
                row=dict(cols=cols,threads=threads,tail=tail,us=ms*1000,registers=exe.n_regs)
                results.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
