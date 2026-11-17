"""Plot the measured per-shape results without rerunning GPU benchmarks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('perf_llir'))
    args = parser.parse_args()
    data = json.loads((args.directory / 'results.json').read_text())
    rows = data['results']
    target = data['metadata'].get('target_speedup', 1.3)
    cols = [r['cols'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    axes[0].plot(cols, [r['api_reference_us'] for r in rows], label='Original Triton')
    axes[0].plot(cols, [r['api_llir_us'] for r in rows], label='Optimized LLVM IR')
    axes[0].set(ylabel='Full-call latency (us)', title='H20 softmax: 4096 rows, float32')
    axes[0].legend()
    axes[1].plot(cols, [r['api_speedup'] for r in rows], label='Full-call speedup')
    axes[1].plot(cols, [r['kernel_speedup'] for r in rows], label='Kernel speedup (CUDA Graph)')
    axes[1].axhline(1, color='gray', linestyle=':')
    axes[1].axhline(target, color='gray', linestyle='--', label=f'{target:.2f}x aggregate target')
    axes[1].set(xlabel='Columns', ylabel='Baseline / optimized time')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.directory / 'performance.png', dpi=160)
    fig.savefig(args.directory / 'performance.svg')


if __name__ == '__main__':
    main()
