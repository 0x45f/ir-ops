"""Explore CUDA-legal warp counts that the Triton frontend normally disallows."""
from pathlib import Path
import sys,types,json
import torch,triton
from edit_softmax_ir import write_ir
from softmax_llir import _launch

# Only LLVM->PTX options need a frontend-compatible placeholder. Actual launch
# metadata and LLVM reqntid retain the requested CUDA thread count.
compiler=types.ModuleType('_experiment_llvm_compiler')
compiler.__file__=str(Path('softmax_llir.py').resolve());sys.modules[compiler.__name__]=compiler
source=Path('softmax_llir.py').read_text().replace('backend.parse_options({"num_warps": num_warps,', 'backend.parse_options({"num_warps": 1 << (num_warps - 1).bit_length(),')
exec(compile(source,compiler.__file__,'exec'),compiler.__dict__)

def main():
 out=Path('perf_nonpower_tuning');out.mkdir(exist_ok=True)
 torch.manual_seed(0);results=[]
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
  for threads in [64,96,128,192,256,320,384,512,640,768,1024]:
   block=(cols+threads*4-1)//(threads*4)*(threads*4)
   p=write_ir(block,threads,directory=out/f't{threads}',tail_only=True)
   exe=compiler._compile_llvm_ir(0,p.read_text(),threads//32,threads//8,1,False)
   fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
   fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
   ms=triton.testing.do_bench(fn,warmup=10,rep=40,return_mode='median')
   row=dict(cols=cols,threads=threads,us=ms*1000,registers=exe.n_regs)
   results.append(row);print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
