"""Compare the Triton and saved-LLVM-IR softmax implementations on one GPU."""

import argparse
import statistics
from pathlib import Path

import torch
import triton

import triton_softmax as reference
from softmax_llir import _compile_llvm_ir, softmax_from_llir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup-ms', type=int, default=100)
    parser.add_argument('--rep-ms', type=int, default=500)
    args = parser.parse_args()
    if min(args.rounds, args.warmup_ms, args.rep_ms) <= 0:
        parser.error('rounds, warmup-ms, and rep-ms must be positive')

    torch.manual_seed(0)
    x = torch.randn((4096, 8320), device=reference.DEVICE, dtype=torch.float32)
    print(f'GPU: {torch.cuda.get_device_name(x.device)}')
    print(f'PyTorch: {torch.__version__}; Triton: {triton.__version__}')
    print(f'Input: shape={tuple(x.shape)}, dtype={x.dtype}, seed=0')
    print(f'Rounds={args.rounds}, warmup={args.warmup_ms} ms, rep={args.rep_ms} ms')

    # Inspect the same launch configuration used by reference.softmax.
    y = torch.empty_like(x)
    kernel = reference.softmax_kernel.warmup(
        y, x, x.stride(0), y.stride(0), *x.shape,
        BLOCK_SIZE=triton.next_power_of_2(x.shape[1]), num_warps=8,
        num_stages=4 if reference.SIZE_SMEM > 200000 else 2, grid=(1,),
    )
    kernel._init_handles()
    occupancy = min(
        reference.NUM_REGS // (kernel.n_regs * reference.WARP_SIZE * 8),
        reference.SIZE_SMEM // kernel.metadata.shared,
    )
    triton_grid = min(reference.NUM_SM * occupancy, x.shape[0])
    ir = Path(__file__).with_name('softmax_kernel.llir').read_text()
    executable = _compile_llvm_ir(x.device.index, ir)
    llir_grid = min(executable.num_programs, x.shape[0])
    print(f'Grid: softmax=({triton_grid}, 1, 1), softmax_from_llir=({llir_grid}, 1, 1)')
    if triton_grid != llir_grid:
        raise RuntimeError('The two implementations have different grids')

    funcs = [('softmax', reference.softmax), ('softmax_from_llir', softmax_from_llir)]
    expected = torch.softmax(x, dim=1)
    for name, fn in funcs:
        actual = fn(x)
        torch.cuda.synchronize(x.device)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        error = (actual - expected).abs().max().item()
        print(f'{name}: correctness PASS, max_abs_error={error:.9g}', flush=True)
        for _ in range(10):
            fn(x)
    torch.cuda.synchronize(x.device)

    # Compilation is complete before timing. Alternate order to reduce order bias.
    # Time the public functions, including their allocation/launch paths, using
    # do_bench CUDA events (not CPU wall-clock timing or CUDA graph replay).
    results = {name: [] for name, _ in funcs}
    for index in range(args.rounds):
        for name, fn in funcs if index % 2 == 0 else funcs[::-1]:
            ms = triton.testing.do_bench(
                lambda: fn(x), warmup=args.warmup_ms, rep=args.rep_ms,
                return_mode='median',
            )
            results[name].append(ms)
        times = ', '.join(f'{name}={values[-1] * 1000:.3f} us' for name, values in results.items())
        print(f'Round {index + 1}: {times}', flush=True)

    print('\nMedian of per-round medians (effective bandwidth = 2 * input bytes / time):')
    medians = {name: statistics.median(values) for name, values in results.items()}
    for name, ms in medians.items():
        gbps = 2 * x.numel() * x.element_size() / (ms * 1e6)
        print(f'{name:20s} {ms * 1000:9.3f} us  {gbps:9.2f} GB/s')
    ratio = medians['softmax_from_llir'] / medians['softmax']
    print(f'LLIR / Triton time: {ratio:.4f}x ({(ratio - 1) * 100:+.2f}%)')


if __name__ == '__main__':
    main()
