"""Explore several rows per CTA with direct LLVM IR."""
from pathlib import Path
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
    out=Path('perf_rows_tuning');out.mkdir(exist_ok=True)
    torch.manual_seed(0);results=[]
    for cols in [1024,2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda'); y=torch.empty_like(x);expected=torch.softmax(x,1)
        for rt,threads in [(32,128),(32,256),(64,128),(64,256),(64,512),(128,128),(128,256),(128,512),(256,256),(256,512)]:
            if rt==32 and cols>4096:continue
            block=(cols+rt*4-1)//(rt*4)*(rt*4)
            p=write_ir(block,threads,directory=out/f'r{rt}_t{threads}',row_threads=rt)
            exe=_compile_llvm_ir(0,p.read_text(),threads//32,0 if rt==32 else threads//8,threads//rt,False)
            fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
            fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
            ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
            row=dict(cols=cols,row_threads=rt,threads=threads,us=ms*1000,registers=exe.n_regs)
            results.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
