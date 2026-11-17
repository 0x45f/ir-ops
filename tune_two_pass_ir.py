"""Test streamed max followed by a cache-reusing second read; full FP32 arithmetic."""
from pathlib import Path
from dataclasses import replace
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
 out=Path('perf_two_pass_tuning');out.mkdir(exist_ok=True)
 torch.manual_seed(0);results=[];sm=torch.cuda.get_device_properties(0).multi_processor_count
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
  for threads in [128,256]:
   for second in [False,True]:
    for grid in [0,4,8]:
     block=(cols+threads*4-1)//(threads*4)*(threads*4)
     p=write_ir(block,threads,directory=out/f't{threads}_two{int(second)}_g{grid}',tail_only=True,two_pass=second,persistent=bool(grid))
     exe=_compile_llvm_ir(0,p.read_text(),threads//32,threads//8,1,bool(grid))
     if grid:exe=replace(exe,num_programs=sm*grid)
     fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
     fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
     ms=triton.testing.do_bench(fn,warmup=15,rep=60,return_mode='median')
     row=dict(cols=cols,threads=threads,two_pass=second,grid=grid,us=ms*1000,registers=exe.n_regs)
     results.append(row);print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
