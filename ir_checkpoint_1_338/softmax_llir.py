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
_BLOCK_SIZE = 16384
_NUM_WARPS = 8
_NUM_STAGES = 4
# Metadata of the IR generated for float32 input of shape (4096, 8320).
_SHARED_BYTES = 196640


@dataclass
class _Executable:
    module: object  # Keep the loaded CUDA module alive while its function is used.
    function: object
    launcher: object
    packed_metadata: tuple
    cubin: bytes
    num_programs: int
    n_regs: int
    rows_per_cta: int = 1


@lru_cache(maxsize=128)
def _compile_llvm_ir(device_index: int, llir: str, num_warps: int = _NUM_WARPS,
                     shared: int = _SHARED_BYTES, rows_per_cta: int = 1,
                     persistent: bool = True) -> _Executable:
    """Run Triton's LLVM IR -> PTX -> cubin stages directly."""
    with torch.cuda.device(device_index):
        target = driver.active.get_current_target()
        if target.backend != "cuda" or target.arch != 90:
            raise RuntimeError("softmax_kernel.llir requires an NVIDIA SM 90 GPU")

        llvm.init_targets()
        backend = CUDABackend(target)
        options = backend.parse_options({"num_warps": num_warps, "num_stages": _NUM_STAGES})
        compile_metadata = {"shared": shared}
        ptx = backend.make_ptx(llir, compile_metadata, options, target.arch)
        cubin = backend.make_cubin(ptx, compile_metadata, options, target.arch)

        # The LLVM IR entry point has six runtime arguments. Triton adds the
        # scratch pointers when loading the binary, as its regular launcher does.
        source = SimpleNamespace(
            signature={0: "*fp32", 1: "*fp32", 2: "i32", 3: "i32", 4: "i32", 5: "i32"},
            constants={},
        )
        launch_metadata = SimpleNamespace(
            num_warps=num_warps,
            num_ctas=1,
            shared=shared,
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
            compile_metadata["name"], cubin, shared, device
        )
        # Match the persistent grid calculation in triton_softmax.softmax.
        properties = driver.active.utils.get_device_properties(device)
        occupancy = properties["max_num_regs"] // (n_regs * properties["warpSize"] * num_warps)
        if shared:
            occupancy = min(occupancy, properties["max_shared_mem"] // shared)
        num_programs = properties["multiprocessor_count"] * occupancy
        return _Executable(module, function, launcher, backend.pack_metadata(launch_metadata),
                           cubin, num_programs if persistent else 0, n_regs, rows_per_cta)


@lru_cache(maxsize=64)
def _get_executable(device_index: int, block: int) -> _Executable:
    path = _LLIR_PATH.parent / 'ir_optimized' / f'softmax_{block}.llir'
    warp_rows = block <= 1024
    return _compile_llvm_ir(device_index, path.read_text(), num_warps=4,
                            shared=0 if warp_rows else 16,
                            rows_per_cta=4 if warp_rows else 1, persistent=False)


def _select_block(cols: int) -> int:
    # LLVM IR need not use power-of-two tiles. Only round to one vector per
    # lane (small rows) or per CTA thread (large rows).
    unit = 128 if cols <= 1024 else 512
    return max(256, (cols + unit - 1) // unit * unit)


def _launch(executable, x, y, stream):
    rows, cols = x.shape
    grid = (rows + executable.rows_per_cta - 1) // executable.rows_per_cta
    if executable.num_programs:
        grid = min(grid, executable.num_programs)
    executable.launcher(
        grid, 1, 1, stream, executable.function, executable.packed_metadata,
        None, None, None, y, x, x.stride(0), y.stride(0), rows, cols,
    )


def softmax_from_llir(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax from the saved LLVM IR on an SM 90 CUDA GPU.

    ``x`` must be contiguous, 2D, float32, and have 1 to 16384 columns.
    Inputs that do not meet the saved IR's alignment specialization are padded.
    Compilation happens once per GPU and column-width bucket. Restart the
    process (or clear both compilation caches) after editing an IR file.
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

    block = _select_block(n_cols)
    device = x.device.index
    executable = _get_executable(device, block)
    if device == driver.active.get_current_device():
        stream = driver.active.get_current_stream(device)
        _launch(executable, x, y, stream)
    else:
        with torch.cuda.device(device):
            stream = driver.active.get_current_stream(device)
            _launch(executable, x, y, stream)
    return y[:, :original_n_cols].contiguous() if needs_padding else y
