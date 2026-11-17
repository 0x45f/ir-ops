"""Exploratory bounded rounding + hardware compression, not public API."""
from pathlib import Path
import json
import torch,triton
from compressed_memory_probe import Buffer,copy,check
from edit_softmax_ir import write_ir
from softmax_llir import _compile_llvm_ir,_launch

def main():
 torch.manual_seed(0);results=[];out=Path('perf_quantized_probe');out.mkdir(exist_ok=True)
 for cols in [2048,4096,6144,8192,8320,12672]:
  x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);buf=Buffer(4096,cols)
  block=(cols+511)//512*512
  for quant in [False,True]:
   p=write_ir(block,128,directory=out/f'q{int(quant)}',tail_only=True,quantize=quant)
   exe=_compile_llvm_ir(0,p.read_text(),4,16,1,False)
   for target in [y,buf]:
    fn=lambda:_launch(exe,x,target,torch.cuda.current_stream().cuda_stream)
    fn()
    if target is buf:check(copy(y.data_ptr(),buf.data_ptr(),x.numel()*4,torch.cuda.current_stream().cuda_stream))
    expected=torch.softmax(x,1)
    torch.testing.assert_close(y,expected,rtol=1e-5,atol=1e-6)
    err=(y-expected).abs().max().item();sums=(y.sum(1)-1).abs().max().item()
    ms=triton.testing.do_bench(fn,warmup=20,rep=100,return_mode='median')
    row=dict(cols=cols,quantized=quant,compressed=target is buf,us=ms*1000,error=err,sum_error=sums)
    results.append(row);print(json.dumps(row),flush=True)
  buf.close()
 (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
