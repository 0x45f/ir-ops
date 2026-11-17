"""Lossless CUDA generic-compression diagnostic; not used by the public API."""
import ctypes as C
import json
from pathlib import Path
import torch,triton
from softmax_llir import _get_executable,_select_block,_launch

class Location(C.Structure):
    _fields_=[('type',C.c_int),('id',C.c_int)]
class Flags(C.Structure):
    _fields_=[('compressionType',C.c_ubyte),('rdma',C.c_ubyte),('usage',C.c_ushort),('reserved',C.c_ubyte*4)]
class Properties(C.Structure):
    _fields_=[('type',C.c_int),('handleTypes',C.c_int),('location',Location),('metadata',C.c_void_p),('flags',Flags)]
class Access(C.Structure):
    _fields_=[('location',Location),('flags',C.c_int)]

cuda=C.CDLL('libcuda.so.1')
def bind(name,args):
    fn=getattr(cuda,name);fn.argtypes=args;fn.restype=C.c_int;return fn
create=bind('cuMemCreate',[C.POINTER(C.c_uint64),C.c_size_t,C.POINTER(Properties),C.c_uint64])
granularity=bind('cuMemGetAllocationGranularity',[C.POINTER(C.c_size_t),C.POINTER(Properties),C.c_int])
reserve=bind('cuMemAddressReserve',[C.POINTER(C.c_uint64),C.c_size_t,C.c_size_t,C.c_uint64,C.c_uint64])
map_memory=bind('cuMemMap',[C.c_uint64,C.c_size_t,C.c_size_t,C.c_uint64,C.c_uint64])
access=bind('cuMemSetAccess',[C.c_uint64,C.c_size_t,C.POINTER(Access),C.c_size_t])
getprops=bind('cuMemGetAllocationPropertiesFromHandle',[C.POINTER(Properties),C.c_uint64])
unmap=bind('cuMemUnmap',[C.c_uint64,C.c_size_t])
release=bind('cuMemRelease',[C.c_uint64])
free_address=bind('cuMemAddressFree',[C.c_uint64,C.c_size_t])
copy=bind('cuMemcpyDtoDAsync_v2',[C.c_uint64,C.c_uint64,C.c_size_t,C.c_void_p])
def check(status):
    if status:raise RuntimeError(f'CUDA status {status}')

class Buffer:
    def __init__(self,rows,cols):
        self.cols=cols
        prop=Properties();prop.type=1;prop.location=Location(1,torch.cuda.current_device());prop.flags.compressionType=1
        gran=C.c_size_t();check(granularity(C.byref(gran),C.byref(prop),1))
        self.size=(rows*cols*4+gran.value-1)//gran.value*gran.value
        self.handle=C.c_uint64();check(create(C.byref(self.handle),self.size,C.byref(prop),0))
        actual=Properties();check(getprops(C.byref(actual),self.handle))
        self.compression=actual.flags.compressionType
        self.address=C.c_uint64();check(reserve(C.byref(self.address),self.size,0,0,0))
        check(map_memory(self.address,self.size,0,self.handle,0))
        desc=Access(prop.location,3);check(access(self.address,self.size,C.byref(desc),1))
    def data_ptr(self):return self.address.value
    def stride(self,axis):return self.cols if axis==0 else 1
    def close(self):
        torch.cuda.synchronize();check(unmap(self.address,self.size));check(release(self.handle));check(free_address(self.address,self.size))

def main():
    torch.manual_seed(0);results=[]
    for cols in [1024,2048,4096,6144,8192,8320,12672]:
        x=torch.randn(4096,cols,device='cuda');y=torch.empty_like(x);buf=Buffer(4096,cols)
        exe=_get_executable(0,_select_block(cols))
        for target in [y,buf]:
            fn=lambda:_launch(exe,x,target,torch.cuda.current_stream().cuda_stream)
            fn()
            if target is buf:check(copy(y.data_ptr(),buf.data_ptr(),x.numel()*4,torch.cuda.current_stream().cuda_stream))
            torch.testing.assert_close(y,torch.softmax(x,1),rtol=1e-5,atol=1e-6)
            ms=triton.testing.do_bench(fn,warmup=20,rep=100,return_mode='median')
            row=dict(cols=cols,compressed=target is buf,supported=buf.compression,us=ms*1000)
            results.append(row);print(json.dumps(row),flush=True)
        buf.close()
    Path('perf_compression_probe.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
