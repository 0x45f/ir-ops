from pathlib import Path
from dataclasses import replace
import json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
 out=Path('perf_prefetch_tuning');out.mkdir(exist_ok=True)
 torch.manual_seed(0);results=[];sm=torch.cuda.get_device_properties(0).multi_processor_count
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
  for prefetch in [False,True]:
   block=(cols+511)//512*512
   p=write_ir(block,128,directory=out/f'prefetch{int(prefetch)}',tail_only=True,persistent=True,prefetch_next=prefetch)
   original=_compile_llvm_ir(0,p.read_text(),4,16,1,True)
   for grid in [2,4,8,16]:
    exe=replace(original,num_programs=sm*grid)
    fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
    fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
    ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
    row=dict(cols=cols,prefetch=prefetch,grid=grid,us=ms*1000,registers=exe.n_regs)
    results.append(row);print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
