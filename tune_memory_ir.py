"""Compare additional load policies; copy is a diagnostic, not a softmax candidate."""
from pathlib import Path
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
    out=Path('perf_memory_tuning');out.mkdir(exist_ok=True)
    torch.manual_seed(0);results=[]
    for cols in [2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x)
        for copy in [False,True]:
            block=(cols+511)//512*512
            p=write_ir(block,128,directory=out/f'copy{int(copy)}',copy_only=copy,tail_only=True)
            text=p.read_text()
            for policy in ['', '.nc', '.L2::128B', '.L2::256B', '.nc.L2::256B']:
                ir=text.replace('ld.global.v4',f'ld.global{policy}.v4')
                exe=_compile_llvm_ir(0,ir,4,16,1,False)
                fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
                fn();torch.testing.assert_close(y,x if copy else torch.softmax(x,1),rtol=1e-5,atol=1e-6)
                ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
                row=dict(cols=cols,copy_only=copy,policy=policy,us=ms*1000,registers=exe.n_regs)
                results.append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
