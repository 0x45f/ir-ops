"""Write LLVM IR text directly; no Triton frontend or computational code generation.

This is a mechanical SSA text editor for the unrolled register-resident rewrite.
The emitted .llir files are the runtime source of truth.
"""
from pathlib import Path


def write_ir(block, threads=128, warp_rows=None, directory='ir_optimized'):
    if warp_rows is None:
        warp_rows = block <= 1024
    assert block % ((32 if warp_rows else threads) * 4) == 0
    row_threads = 32 if warp_rows else threads
    lines = []
    emit = lines.append
    counter = 0

    def ins(text):
        nonlocal counter
        counter += 1
        name = f'%v{counter}'
        emit(f'  {name} = {text}')
        return name

    def reduce(values, op):
        while len(values) > 1:
            values = [combine(values[i], values[i+1], op) if i+1 < len(values) else values[i]
                      for i in range(0, len(values), 2)]
        return values[0]

    def combine(a, b, op):
        return ins(f'call float @llvm.maxnum.f32(float {a}, float {b})' if op == 'max'
                   else f'fadd float {a}, {b}')

    def shuffle(value, op):
        for distance in [16, 8, 4, 2, 1]:
            bits = ins(f'bitcast float {value} to i32')
            other = ins(f'call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {bits}, i32 {distance}, i32 31)')
            other = ins(f'bitcast i32 {other} to float')
            value = combine(value, other, op)
        return value

    def row_reduce(value, op):
        value = shuffle(value, op)
        if not warp_rows:
            emit(f'  call void asm sideeffect "@$2 st.shared.b32 [$0], $1;", "r,f,b"(ptr addrspace(3) {shared_ptr}, float {value}, i1 {leader})')
            emit('  call void @llvm.nvvm.barrier0()')
            values = [ins(f'load volatile float, ptr addrspace(3) getelementptr (float, ptr addrspace(3) @global_smem, i32 {i}), align 4') for i in range(threads // 32)]
            value = reduce(values, op)
            # All warps must finish reading the max before any writes the sum.
            emit('  call void @llvm.nvvm.barrier0()')
        return value

    emit('; Direct LLVM IR rewrite: vector loads -> register reductions -> vector stores.')
    emit(f'; block={block}, num_warps={threads // 32}, rows_per_cta={threads // 32 if warp_rows else 1}, shared={0 if warp_rows else threads // 8}')
    emit('target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"')
    emit('@global_smem = external addrspace(3) global [0 x i8], align 16')
    emit('define ptx_kernel void @softmax_kernel(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %istride, i32 %ostride, i32 %rows, i32 %cols, ptr addrspace(1) %scratch, ptr addrspace(1) %profile) #0 {')
    emit('entry:')
    tid = ins('call i32 @llvm.nvvm.read.ptx.sreg.tid.x()')
    cta = ins('call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()')
    lane = ins(f'and i32 {tid}, 31')
    warp = ins(f'lshr i32 {tid}, 5')
    if warp_rows:
        row = ins(f'mul i32 {cta}, {threads // 32}')
        row = ins(f'add i32 {row}, {warp}')
    else:
        row = cta
    valid = ins(f'icmp ult i32 {row}, %rows')
    emit(f'  br i1 {valid}, label %body, label %exit')
    emit('body:')
    row64 = ins(f'zext i32 {row} to i64')
    istride = ins('zext i32 %istride to i64')
    ostride = ins('zext i32 %ostride to i64')
    off = ins(f'mul i64 {row64}, {istride}')
    ip = ins(f'getelementptr float, ptr addrspace(1) %in, i64 {off}')
    off = ins(f'mul i64 {row64}, {ostride}')
    optr = ins(f'getelementptr float, ptr addrspace(1) %out, i64 {off}')
    col = ins(f'mul i32 {lane if warp_rows else tid}, 4')
    if not warp_rows:
        shared_ptr = ins(f'getelementptr float, ptr addrspace(3) @global_smem, i32 {warp}')
        leader = ins(f'icmp eq i32 {lane}, 0')
    vals, groups = [], []
    for k in range(block // (row_threads * 4)):
        idx = ins(f'add i32 {col}, {k * row_threads * 4}')
        pred = ins(f'icmp ult i32 {idx}, %cols')
        idx64 = ins(f'zext i32 {idx} to i64')
        p = ins(f'getelementptr float, ptr addrspace(1) {ip}, i64 {idx64}')
        vec = ins('call {float, float, float, float} asm sideeffect '
                  '"mov.b32 $0, 0xff800000; mov.b32 $1, 0xff800000; mov.b32 $2, 0xff800000; mov.b32 $3, 0xff800000; @$5 ld.global.v4.b32 {$0,$1,$2,$3}, [$4];", '
                  f'"=f,=f,=f,=f,l,b"(ptr addrspace(1) {p}, i1 {pred})')
        vals.extend(ins(f'extractvalue {{float, float, float, float}} {vec}, {j}') for j in range(4))
        groups.append((idx64, pred))
    maximum = row_reduce(reduce(vals, 'max'), 'max')
    exps = []
    for value in vals:
        diff = ins(f'fsub float {value}, {maximum}')
        scaled = ins(f'fmul float {diff}, 0x3FF7154760000000')
        exps.append(ins(f'call float @llvm.nvvm.ex2.approx.f32(float {scaled})'))
    total = row_reduce(reduce(exps, 'sum'), 'sum')
    inverse = ins(f'call float asm "rcp.approx.ftz.f32 $0, $1;", "=f,f"(float {total})')
    for k, (idx64, pred) in enumerate(groups):
        p = ins(f'getelementptr float, ptr addrspace(1) {optr}, i64 {idx64}')
        ys = [ins(f'fmul float {value}, {inverse}') for value in exps[k*4:k*4+4]]
        args = ', '.join(f'float {v}' for v in ys)
        emit(f'  call void asm sideeffect "@$5 st.global.v4.b32 [$4], {{$0,$1,$2,$3}};", "f,f,f,f,l,b"({args}, ptr addrspace(1) {p}, i1 {pred})')
    emit('  br label %exit')
    emit('exit:\n  ret void\n}')
    emit('declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()')
    emit('declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()')
    emit('declare float @llvm.maxnum.f32(float, float)')
    emit('declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)')
    emit('declare float @llvm.nvvm.ex2.approx.f32(float)')
    emit('declare void @llvm.nvvm.barrier0()')
    emit(f'attributes #0 = {{ nounwind "nvvm.reqntid"="{threads}" }}')
    path = Path(directory) / f'softmax_{block}.llir'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return path


if __name__ == '__main__':
    for block in [*range(256, 1025, 128), *range(1536, 16385, 512)]:
        write_ir(block)
