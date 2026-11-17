"""Paired full-call timing and CUDA-graph kernel timing over the reference shapes."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import triton

import triton_softmax as reference
from softmax_llir import _get_executable, _launch, _select_block, softmax_from_llir


def reference_kernel(x, y):
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    stages = 4 if reference.SIZE_SMEM > 200000 else 2
    kernel = reference.softmax_kernel.warmup(
        y, x, cols, cols, rows, cols, BLOCK_SIZE=block,
        num_stages=stages, num_warps=8, grid=(1,),
    )
    kernel._init_handles()
    occupancy = min(reference.NUM_REGS // (kernel.n_regs * reference.WARP_SIZE * 8),
                    reference.SIZE_SMEM // kernel.metadata.shared)
    grid = min(reference.NUM_SM * occupancy, rows)
    return (lambda: kernel[(grid, 1, 1)](y, x, cols, cols, rows, cols, block, stages)), kernel, grid


def speedup_ci(baseline, optimized):
    # Paired-round bootstrap; intervals describe measurement variability only.
    logs = np.log(np.array(baseline) / np.array(optimized))
    rng = np.random.default_rng(0)
    samples = rng.choice(logs, size=(10000, len(logs)), replace=True).mean(axis=1)
    return np.exp(np.quantile(samples, [0.025, 0.975])).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmup-ms', type=int, default=100)
    parser.add_argument('--rep-ms', type=int, default=500)
    parser.add_argument('--kernel-rep-ms', type=int, default=100)
    parser.add_argument('--cols', type=int, nargs='+', default=list(range(256, 12800, 128)))
    parser.add_argument('--output', type=Path, default=Path('perf_llir'))
    args = parser.parse_args()
    if min(args.rounds, args.warmup_ms, args.rep_ms, args.kernel_rep_ms) <= 0:
        parser.error('timing parameters must be positive')
    if any(not 16 <= n <= 16384 or n % 16 for n in args.cols):
        parser.error('benchmark columns must be multiples of 16 between 16 and 16384')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    rounds=args.rounds, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                    kernel_rep_ms=args.kernel_rep_ms, seed=0,
                    api_timing='do_bench CUDA events, allocations included, compilation excluded',
                    kernel_timing='do_bench_cudagraph, preallocated output, graph replay',
                    tolerance=dict(rtol=1e-5, atol=1e-6))
    root = Path(__file__).parent
    sources = [root / name for name in ['triton_softmax.py', 'softmax_llir.py', 'benchmark_softmax_llir.py']]
    sources += sorted((root / 'ir_optimized').glob('*.llir'))
    metadata['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    print(json.dumps({k: v for k, v in metadata.items() if k != 'source_sha256'}), flush=True)
    results = []
    for cols in args.cols:
        x = torch.randn((4096, cols), device=reference.DEVICE, dtype=torch.float32)
        y_ref, y_ir = torch.empty_like(x), torch.empty_like(x)
        ref_launch, kernel, ref_grid = reference_kernel(x, y_ref)
        block = _select_block(cols)
        executable = _get_executable(x.device.index, block)
        # Graph timing captures on a side stream: resolve the stream at launch,
        # never close over the stream that was current before capture.
        ir_launch = lambda: _launch(executable, x, y_ir, torch.cuda.current_stream().cuda_stream)
        expected = torch.softmax(x, dim=1)
        # Both public and isolated paths must pass before measurements.
        errors = {}
        for name, fn in [('reference', reference.softmax), ('llir', softmax_from_llir)]:
            actual = fn(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            errors[name] = (actual - expected).abs().max().item()
        ref_launch()
        ir_launch()
        torch.testing.assert_close(y_ref, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(y_ir, expected, rtol=1e-5, atol=1e-6)
        del expected, actual
        functions = dict(reference=lambda: reference.softmax(x), llir=lambda: softmax_from_llir(x))
        launches = dict(reference=ref_launch, llir=ir_launch)
        timings = {kind: {name: [] for name in functions} for kind in ['api', 'kernel']}
        for round_index in range(args.rounds):
            order = ['reference', 'llir'] if round_index % 2 == 0 else ['llir', 'reference']
            for name in order:
                ms = triton.testing.do_bench(functions[name], warmup=args.warmup_ms,
                                            rep=args.rep_ms, return_mode='median')
                timings['api'][name].append(ms * 1000)
            for name in order:
                ms = triton.testing.do_bench_cudagraph(launches[name], rep=args.kernel_rep_ms,
                                                      return_mode='median')
                timings['kernel'][name].append(ms * 1000)
        row = dict(rows=4096, cols=cols, max_abs_error=errors['llir'],
                   reference_grid=ref_grid, llir_grid=(4096 + executable.rows_per_cta - 1) // executable.rows_per_cta,
                   reference_registers=kernel.n_regs, llir_registers=executable.n_regs,
                   reference_shared=kernel.metadata.shared, llir_shared=executable.packed_metadata[2],
                   rounds=timings)
        for kind, times in timings.items():
            for name, values in times.items():
                row[f'{kind}_{name}_us'] = statistics.median(values)
            row[f'{kind}_speedup'] = row[f'{kind}_reference_us'] / row[f'{kind}_llir_us']
            row[f'{kind}_speedup_ci95'] = speedup_ci(times['reference'], times['llir'])
        results.append(row)
        print(f'N={cols:5d} API {row["api_reference_us"]:.2f} -> {row["api_llir_us"]:.2f} us '
              f'({row["api_speedup"]:.3f}x); kernel {row["kernel_reference_us"]:.2f} -> '
              f'{row["kernel_llir_us"]:.2f} us ({row["kernel_speedup"]:.3f}x)', flush=True)
        (args.output / 'results.json').write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + '\n')
    summary = dict(shapes=len(results))
    for kind in ['api', 'kernel']:
        summary[f'{kind}_geomean_speedup'] = math.exp(statistics.mean(math.log(r[f'{kind}_speedup']) for r in results))
        summary[f'{kind}_minimum_speedup'] = min(r[f'{kind}_speedup'] for r in results)
        summary[f'{kind}_significant_regressions'] = [r['cols'] for r in results if r[f'{kind}_speedup_ci95'][1] < 1]
    summary['full_shape_coverage'] = sorted(args.cols) == list(range(256, 12800, 128))
    summary['accepted'] = (summary['full_shape_coverage'] and args.rounds >= 5 and
                           summary['api_geomean_speedup'] >= 1.30 and
                           not summary['api_significant_regressions'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = [k for k in results[0] if k != 'rounds']
    with (args.output / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in results)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
