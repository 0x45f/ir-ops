"""Export the UNMODIFIED reference kernel's LLVM IR for every benchmark shape."""
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton_softmax as reference


def main():
    directory = Path(__file__).with_name('ir_baseline')
    directory.mkdir(exist_ok=True)
    manifest = []
    for cols in range(256, 12800, 128):
        x = torch.empty((4096, cols), device='cuda', dtype=torch.float32)
        y = torch.empty_like(x)
        kernel = reference.softmax_kernel.warmup(
            y, x, cols, cols, 4096, cols, BLOCK_SIZE=triton.next_power_of_2(cols),
            num_stages=4 if reference.SIZE_SMEM > 200000 else 2,
            num_warps=8, grid=(1,),
        )
        kernel._init_handles()
        ir = kernel.asm['llir']
        digest = hashlib.sha256(ir.encode()).hexdigest()
        name = f'{digest[:16]}.llir'
        (directory / name).write_text(ir)
        manifest.append(dict(rows=4096, cols=cols, block=triton.next_power_of_2(cols),
                             file=name, sha256=digest, registers=kernel.n_regs,
                             shared=kernel.metadata.shared))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest)} shapes, {len(set(r["file"] for r in manifest))} distinct IR modules')


if __name__ == '__main__':
    main()
