"""Compile and run the saved LLVM IR for the float32 softmax kernel."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from triton._C.libtriton import llvm
from triton.backends.nvidia.compiler import CUDABackend
from triton.runtime.driver import driver


_LLIR_PATH = Path(__file__).with_name("softmax_kernel.llir")
_BLOCK_SIZE = 8192
_NUM_WARPS = 8
_NUM_STAGES = 4
# Metadata of the IR generated for float32 input of shape (4096, 8064).
_SHARED_BYTES = 98336


@dataclass
class _Executable:
    module: object  # Keep the loaded CUDA module alive while its function is used.
    function: object
    launcher: object
    packed_metadata: tuple
    cubin: bytes
    num_programs: int


@lru_cache(maxsize=8)
def _compile_llvm_ir(device_index: int, llir: str) -> _Executable:
    """Run Triton's LLVM IR -> PTX -> cubin stages directly."""
    with torch.cuda.device(device_index):
        target = driver.active.get_current_target()
        if target.backend != "cuda" or target.arch != 90:
            raise RuntimeError("softmax_kernel.llir requires an NVIDIA SM 90 GPU")

        llvm.init_targets()
        backend = CUDABackend(target)
        options = backend.parse_options({"num_warps": _NUM_WARPS, "num_stages": _NUM_STAGES})
        compile_metadata = {"shared": _SHARED_BYTES}
        ptx = backend.make_ptx(llir, compile_metadata, options, target.arch)
        cubin = backend.make_cubin(ptx, compile_metadata, options, target.arch)

        # The LLVM IR entry point has six runtime arguments. Triton adds the
        # scratch pointers when loading the binary, as its regular launcher does.
        source = SimpleNamespace(
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32", 4: "i32", 5: "i32"},
            constants={},
        )
        launch_metadata = SimpleNamespace(
            num_warps=_NUM_WARPS,
            num_ctas=1,
            shared=_SHARED_BYTES,
            global_scratch_size=0,
            global_scratch_align=1,
            profile_scratch_size=0,
            profile_scratch_align=1,
            launch_cooperative_grid=False,
            launch_pdl=False,
        )
        launcher = driver.active.launcher_cls(source, launch_metadata)
        device = driver.active.get_current_device()
        module, function, n_regs, _, _ = driver.active.utils.load_binary(
            compile_metadata["name"], cubin, _SHARED_BYTES, device
        )
        # Match the persistent grid calculation in triton_softmax.softmax.
        properties = driver.active.utils.get_device_properties(device)
        occupancy = properties["max_num_regs"] // (n_regs * properties["warpSize"] * _NUM_WARPS)
        occupancy = min(occupancy, properties["max_shared_mem"] // _SHARED_BYTES)
        num_programs = properties["multiprocessor_count"] * occupancy
        return _Executable(module, function, launcher, backend.pack_metadata(launch_metadata), cubin, num_programs)


def softmax_from_llir(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax from the saved LLVM IR on an SM 90 CUDA GPU.

    ``x`` must be contiguous, 2D, float32, and have 1 to 8192 columns.
    Inputs that do not meet the saved IR's alignment specialization are padded.
    Compilation happens on the first call for each GPU and IR content.
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    n_rows, n_cols = x.shape
    if not 1 <= n_cols <= _BLOCK_SIZE:
        raise ValueError(f"x must have between 1 and {_BLOCK_SIZE} columns")
    if not _LLIR_PATH.is_file():
        raise FileNotFoundError(_LLIR_PATH)

    if n_rows == 0:
        return torch.empty_like(x)

    # The saved IR specializes row strides/column count to multiples of 16
    # and uses aligned 16-byte vector accesses. Pad with -inf so these
    # assumptions hold without changing the softmax of the original columns.
    original_n_cols = n_cols
    needs_padding = n_cols % 16 != 0 or x.data_ptr() % 16 != 0
    if needs_padding:
        n_cols = (n_cols + 15) // 16 * 16
        padded = torch.full((n_rows, n_cols), -float("inf"), device=x.device, dtype=x.dtype)
        padded[:, :original_n_cols].copy_(x)
        x = padded
    y = torch.empty_like(x)

    with torch.cuda.device(x.device):
        executable = _compile_llvm_ir(x.device.index, _LLIR_PATH.read_text())
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        executable.launcher(
            min(executable.num_programs, n_rows), 1, 1, stream, executable.function, executable.packed_metadata,
            None, None, None, y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        )
    return y[:, :original_n_cols].contiguous() if needs_padding else y
