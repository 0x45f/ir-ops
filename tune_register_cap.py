"""Compare ptxas register caps for the same saved LLVM IR."""
import sys,types,json
from pathlib import Path
import torch,triton
from softmax_llir import _launch,_select_block


def make_compiler(cap):
 module=types.ModuleType(f'_register_cap_{cap}');module.__file__=str(Path('softmax_llir.py').resolve())
 sys.modules[module.__name__]=module
 source=Path('softmax_llir.py').read_text().replace('"num_stages": _NUM_STAGES}',f'"num_stages": _NUM_STAGES, "ptx_options": "--maxrregcount={cap}"}}')
 exec(compile(source,module.__file__,'exec'),module.__dict__)
 return module

def main():
 out=Path('perf_register_tuning');out.mkdir(exist_ok=True)
 torch.manual_seed(0);results=[]
 compilers={cap:make_compiler(cap) for cap in [32,48,64,80,96,128,255]}
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
  ir=Path(f'ir_optimized/softmax_{_select_block(cols)}.llir').read_text()
  for cap,module in compilers.items():
   exe=module._compile_llvm_ir(0,ir,4,16,1,False)
   fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
   fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
   ms=triton.testing.do_bench(fn,warmup=15,rep=60,return_mode='median')
   row=dict(cols=cols,cap=cap,actual_registers=exe.n_regs,us=ms*1000)
   results.append(row);print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
