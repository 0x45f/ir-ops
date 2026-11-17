"""Effective register caps embedded directly in LLVM IR function attributes."""
from pathlib import Path
import json
import torch,triton
from softmax_llir import _compile_llvm_ir,_launch,_select_block

def main():
 out=Path('perf_register_ir_tuning');out.mkdir(exist_ok=True)
 torch.manual_seed(0);results=[]
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);expected=torch.softmax(x,1)
  original=Path(f'ir_optimized/softmax_{_select_block(cols)}.llir').read_text()
  for cap in [32,48,64,80,96,128]:
   ir=original.replace('"nvvm.reqntid"="128"',f'"nvvm.reqntid"="128" "nvvm.maxnreg"="{cap}"')
   (out/f'n{cols}_r{cap}.llir').write_text(ir)
   exe=_compile_llvm_ir(0,ir,4,16,1,False)
   fn=lambda:_launch(exe,x,y,torch.cuda.current_stream().cuda_stream)
   fn();torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
   ms=triton.testing.do_bench(fn,warmup=15,rep=60,return_mode='median')
   row=dict(cols=cols,cap=cap,actual_registers=exe.n_regs,us=ms*1000)
   results.append(row);print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
