"""Direct LLVM IR bulk TMA store experiment."""
from pathlib import Path
import ctypes,json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
    out=Path('perf_tma_tuning');out.mkdir(exist_ok=True)
    cuda=ctypes.CDLL('libcuda.so.1')
    config=cuda.cuFuncSetCacheConfig;config.argtypes=[ctypes.c_void_p,ctypes.c_int]
    torch.manual_seed(0);results=[]
    for cols in [2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
        for threads in [128,256]:
            block=(cols+threads*4-1)//(threads*4)*(threads*4)
            for tma in [False,True]:
                p=write_ir(block,threads,directory=out/f't{threads}_tma{int(tma)}',tail_only=True,tma_store=tma)
                shared=block*4+16 if tma else threads//8
                exe=_compile_llvm_ir(0,p.read_text(),threads//32,shared,1,False)
                assert config(exe.function,1)==0
                fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
                fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
                ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
                row=dict(cols=cols,threads=threads,tma=tma,us=ms*1000,registers=exe.n_regs)
                results.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
