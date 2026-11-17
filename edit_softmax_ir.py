"""Write LLVM IR text directly; no Triton frontend or computational code generation.

This is a mechanical SSA text editor for the unrolled register-resident rewrite.
The emitted .llir files are the runtime source of truth.
"""
from pathlib import Path


def write_ir(block, threads=128, warp_rows=None, directory='ir_optimized', persistent=False,
             row_threads=None, fast_exp=False, tail_only=False, rotate=False, redux=False,
             copy_only=False, tma_store=False, tma_load=False, quantize=False,
             prefetch_next=False, separate_scratch=False, two_pass=False):
    """Mechanically edit LLVM SSA for a tile and optional experimental variants.

    With tail_only, the caller must select the smallest tile rounded up to
    row_threads*4 (except the minimum 256-column tile, which stays fully masked).
    The public deployment uses only tail_only; the other switches preserve
    independent, rejected/diagnostic experiments and are never selected at runtime.
    """
    if warp_rows is None:
        warp_rows = block <= 1024
    row_threads = row_threads or (32 if warp_rows else threads)
    warp_rows = row_threads == 32
    rows_per_cta = threads // row_threads
    assert threads % row_threads == 0 and row_threads % 32 == 0
    assert block % (row_threads * 4) == 0
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
        if op == 'max' and redux:
            bits = ins(f'bitcast float {value} to i32')
            sign = ins(f'ashr i32 {bits}, 31')
            mask = ins(f'and i32 {sign}, 2147483647')
            ordered = ins(f'xor i32 {bits}, {mask}')
            maximum = ins(f'call i32 asm "redux.sync.max.s32 $0, $1, -1;", "=r,r"(i32 {ordered})')
            sign = ins(f'ashr i32 {maximum}, 31')
            mask = ins(f'and i32 {sign}, 2147483647')
            bits = ins(f'xor i32 {maximum}, {mask}')
            return ins(f'bitcast i32 {bits} to float')
        for distance in [16, 8, 4, 2, 1]:
            bits = ins(f'bitcast float {value} to i32')
            other = ins(f'call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {bits}, i32 {distance}, i32 31)')
            other = ins(f'bitcast i32 {other} to float')
            value = combine(value, other, op)
        return value

    def row_reduce(value, op):
        value = shuffle(value, op)
        if not warp_rows:
            base, write_ptr = shared_base, shared_ptr
            if separate_scratch and op == 'sum':
                assert not tma_store and not tma_load
                idx = ins(f'add i32 {warp}, {threads // 32}')
                write_ptr = ins(f'getelementptr float, ptr addrspace(3) @global_smem, i32 {idx}')
                base = ins(f'add i32 {shared_base}, {threads // 32}')
            emit(f'  call void asm sideeffect "@$2 st.shared.b32 [$0], $1;", "r,f,b"(ptr addrspace(3) {write_ptr}, float {value}, i1 {leader})')
            emit('  call void @llvm.nvvm.barrier0()')
            values = []
            for i in range(row_threads // 32):
                idx = ins(f'add i32 {base}, {i}')
                ptr = ins(f'getelementptr float, ptr addrspace(3) @global_smem, i32 {idx}')
                values.append(ins(f'load volatile float, ptr addrspace(3) {ptr}, align 4'))
            value = reduce(values, op)
            # All warps must finish reading the max before any writes the sum.
            if not separate_scratch:
                emit('  call void @llvm.nvvm.barrier0()')
        return value

    emit('; Direct LLVM IR rewrite: vector loads -> register reductions -> vector stores.')
    emit(f'; block={block}, num_warps={threads // 32}, rows_per_cta={rows_per_cta}, shared={0 if warp_rows else threads // 8}')
    emit('target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"')
    emit('@global_smem = external addrspace(3) global [0 x i8], align 16')
    emit('define ptx_kernel void @softmax_kernel(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %istride, i32 %ostride, i32 %rows, i32 %cols, ptr addrspace(1) %scratch, ptr addrspace(1) %profile) #0 {')
    emit('entry:')
    tid = ins('call i32 @llvm.nvvm.read.ptx.sreg.tid.x()')
    cta = ins('call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()')
    lane = ins(f'and i32 {tid}, 31')
    warp = ins(f'lshr i32 {tid}, 5')
    row_in_cta = ins(f'udiv i32 {tid}, {row_threads}')
    if rows_per_cta > 1:
        first_row = ins(f'mul i32 {cta}, {rows_per_cta}')
        row = ins(f'add i32 {first_row}, {row_in_cta}')
    else:
        first_row = cta
        row = cta
    if persistent:
        grid = ins('call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()')
        assert rows_per_cta == 1 or warp_rows
        step = ins(f'mul i32 {grid}, {rows_per_cta}')
        emit('  br label %row_loop')
        emit('row_loop:')
        emit(f'  %row = phi i32 [{row}, %entry], [%nextrow, %advance]')
        row = '%row'
    valid = ins(f'icmp ult i32 {row}, %rows')
    active = valid if rows_per_cta == 1 or warp_rows else ins(f'icmp ult i32 {first_row}, %rows')
    emit(f'  br i1 {active}, label %body, label %exit')
    emit('body:')
    row64 = ins(f'zext i32 {row} to i64')
    istride = ins('zext i32 %istride to i64')
    ostride = ins('zext i32 %ostride to i64')
    off = ins(f'mul i64 {row64}, {istride}')
    ip = ins(f'getelementptr float, ptr addrspace(1) %in, i64 {off}')
    off = ins(f'mul i64 {row64}, {ostride}')
    optr = ins(f'getelementptr float, ptr addrspace(1) %out, i64 {off}')
    row_tid = ins(f'urem i32 {tid}, {row_threads}')
    col = ins(f'mul i32 {row_tid}, 4')
    if not warp_rows:
        shared_base = ins(f'mul i32 {row_in_cta}, {row_threads // 32}')
        shared_ptr = ins(f'getelementptr float, ptr addrspace(3) @global_smem, i32 {warp}')
        leader = ins(f'icmp eq i32 {lane}, 0')
    if tma_load:
        assert rows_per_cta == 1 and not fast_exp and not persistent
        tma_buffer = ins('getelementptr i8, ptr addrspace(3) @global_smem, i32 128')
        tma_barrier = ins('getelementptr i8, ptr addrspace(3) @global_smem, i32 64')
        tma_bytes = ins('mul i32 %cols, 4')
        emit('  call void asm sideeffect "{ .reg .pred p; setp.eq.u32 p, $4, 0; '
             '@p mbarrier.init.shared::cta.b64 [$2], 1; '
             '@p fence.proxy.async.shared::cta; '
             '@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$2], $3; '
             '@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2]; }", '
             f'"r,l,r,r,r"(ptr addrspace(3) {tma_buffer}, ptr addrspace(1) {ip}, ptr addrspace(3) {tma_barrier}, i32 {tma_bytes}, i32 {tid})')
        emit('  call void @llvm.nvvm.barrier0()')
        emit('  call void asm sideeffect "{ .reg .pred p; wait_copy: '
             'mbarrier.try_wait.parity.shared::cta.b64 p, [$0], 0; @!p bra wait_copy; }", '
             f'"r"(ptr addrspace(3) {tma_barrier})')
    if fast_exp:
        assert rows_per_cta == 1 or warp_rows
        fast_values, fast_groups = [], []
        for k in range(block // (row_threads * 4)):
            idx = ins(f'add i32 {col}, {k * row_threads * 4}')
            pred = ins(f'icmp ult i32 {idx}, %cols')
            idx64 = ins(f'zext i32 {idx} to i64')
            p = ins(f'getelementptr float, ptr addrspace(1) {ip}, i64 {idx64}')
            vec = ins('call {float, float, float, float} asm sideeffect '
                      '"mov.b32 $0, 0xff800000; mov.b32 $1, 0xff800000; mov.b32 $2, 0xff800000; mov.b32 $3, 0xff800000; @$5 ld.global.v4.b32 {$0,$1,$2,$3}, [$4];", '
                      f'"=f,=f,=f,=f,l,b"(ptr addrspace(1) {p}, i1 {pred})')
            for j in range(4):
                value = ins(f'extractvalue {{float, float, float, float}} {vec}, {j}')
                scaled = ins(f'fmul float {value}, 0x3FF7154760000000')
                fast_values.append(ins(f'call float @llvm.nvvm.ex2.approx.f32(float {scaled})'))
            fast_groups.append((idx64, pred))
        total = row_reduce(reduce(fast_values, 'sum'), 'sum')
        low = ins(f'fcmp ogt float {total}, 0x3DF0000000000000')
        high = ins(f'fcmp olt float {total}, 0x41F0000000000000')
        safe = ins(f'and i1 {low}, {high}')
        emit(f'  br i1 {safe}, label %fast_store, label %stable')
        emit('fast_store:')
        inverse = ins(f'call float asm "rcp.approx.ftz.f32 $0, $1;", "=f,f"(float {total})')
        for k, (idx64, pred) in enumerate(fast_groups):
            p = ins(f'getelementptr float, ptr addrspace(1) {optr}, i64 {idx64}')
            ys = [ins(f'fmul float {value}, {inverse}') for value in fast_values[k*4:k*4+4]]
            args = ', '.join(f'float {v}' for v in ys)
            emit(f'  call void asm sideeffect "@$5 st.global.v4.b32 [$4], {{$0,$1,$2,$3}};", "f,f,f,f,l,b"({args}, ptr addrspace(1) {p}, i1 {pred})')
        emit(f'  br label %{ "advance" if persistent else "exit" }')
        emit('stable:')
    if two_pass:
        assert not any((fast_exp, rotate, tma_load, tma_store, copy_only, quantize))
        local_max = None
        for k in range(block // (row_threads * 4)):
            idx = ins(f'add i32 {col}, {k * row_threads * 4}')
            pred = ins(f'icmp ult i32 {idx}, %cols')
            idx64 = ins(f'zext i32 {idx} to i64')
            p = ins(f'getelementptr float, ptr addrspace(1) {ip}, i64 {idx64}')
            vec = ins('call {float, float, float, float} asm sideeffect '
                      '"mov.b32 $0, 0xff800000; mov.b32 $1, 0xff800000; mov.b32 $2, 0xff800000; mov.b32 $3, 0xff800000; @$5 ld.global.ca.v4.b32 {$0,$1,$2,$3}, [$4];", '
                      f'"=f,=f,=f,=f,l,b"(ptr addrspace(1) {p}, i1 {pred})')
            chunk = [ins(f'extractvalue {{float, float, float, float}} {vec}, {j}') for j in range(4)]
            chunk_max = reduce(chunk, 'max')
            local_max = chunk_max if local_max is None else combine(local_max, chunk_max, 'max')
        stream_max = row_reduce(local_max, 'max')
    vals, groups = [], []
    if rotate:
        assert not tail_only
        rotation = ins(f'urem i32 {row}, {block // (row_threads * 4)}')
    for k in range(block // (row_threads * 4)):
        if rotate:
            group = ins(f'add i32 {rotation}, {k}')
            group = ins(f'urem i32 {group}, {block // (row_threads * 4)}')
            offset = ins(f'mul i32 {group}, {row_threads * 4}')
            idx = ins(f'add i32 {col}, {offset}')
        else:
            idx = ins(f'add i32 {col}, {k * row_threads * 4}')
        full = tail_only and block > 256 and k < block // (row_threads * 4) - 1
        pred = 'true' if full else ins(f'icmp ult i32 {idx}, %cols')
        if rows_per_cta > 1 and not warp_rows:
            pred = ins(f'and i1 {pred}, {valid}')
        idx64 = ins(f'zext i32 {idx} to i64')
        p = ins(f'getelementptr float, ptr addrspace({3 if tma_load else 1}) {tma_buffer if tma_load else ip}, i64 {idx64}')
        memory = 'shared' if tma_load else 'global'
        space, constraint = (3, 'r') if tma_load else (1, 'l')
        if full and (rows_per_cta == 1 or warp_rows):
            vec = ins('call {float, float, float, float} asm sideeffect '
                      f'"ld.{memory}.v4.b32 {{$0,$1,$2,$3}}, [$4];", '
                      f'"=f,=f,=f,=f,{constraint}"(ptr addrspace({space}) {p})')
        else:
            vec = ins('call {float, float, float, float} asm sideeffect '
                  f'"mov.b32 $0, 0xff800000; mov.b32 $1, 0xff800000; mov.b32 $2, 0xff800000; mov.b32 $3, 0xff800000; @$5 ld.{memory}.v4.b32 {{$0,$1,$2,$3}}, [$4];", '
                  f'"=f,=f,=f,=f,{constraint},b"(ptr addrspace({space}) {p}, i1 {pred})')
        chunk = [ins(f'extractvalue {{float, float, float, float}} {vec}, {j}') for j in range(4)]
        if two_pass:
            for value in chunk:
                diff = ins(f'fsub float {value}, {stream_max}')
                scaled = ins(f'fmul float {diff}, 0x3FF7154760000000')
                vals.append(ins(f'call float @llvm.nvvm.ex2.approx.f32(float {scaled})'))
        else:
            vals.extend(chunk)
        groups.append((idx64, pred))
    if prefetch_next:
        assert persistent and rows_per_cta == 1
        next_row = ins(f'add i32 {row}, {step}')
        next_valid = ins(f'icmp ult i32 {next_row}, %rows')
        next64 = ins(f'zext i32 {next_row} to i64')
        next_offset = ins(f'mul i64 {next64}, {istride}')
        next_ptr = ins(f'getelementptr float, ptr addrspace(1) %in, i64 {next_offset}')
        sublane = ins(f'and i32 {tid}, 7')
        first = ins(f'icmp eq i32 {sublane}, 0')
        enabled = ins(f'and i1 {first}, {next_valid}')
        for index, valid_col in groups:
            address = ins(f'getelementptr float, ptr addrspace(1) {next_ptr}, i64 {index}')
            pred = ins(f'and i1 {enabled}, {valid_col}')
            emit(f'  call void asm sideeffect "@$1 prefetch.global.L2 [$0];", "l,b"(ptr addrspace(1) {address}, i1 {pred})')
    if copy_only:
        exps, inverse = vals, '1.000000e+00'
    else:
        if two_pass:
            exps = vals
        else:
            maximum = row_reduce(reduce(vals, 'max'), 'max')
            exps = []
            for value in vals:
                diff = ins(f'fsub float {value}, {maximum}')
                scaled = ins(f'fmul float {diff}, 0x3FF7154760000000')
                exps.append(ins(f'call float @llvm.nvvm.ex2.approx.f32(float {scaled})'))
        total = row_reduce(reduce(exps, 'sum'), 'sum')
        inverse = ins(f'call float asm "rcp.approx.ftz.f32 $0, $1;", "=f,f"(float {total})')
    for k, (idx64, pred) in enumerate(groups):
        if tma_store:
            assert rows_per_cta == 1 and not fast_exp
            smem_out = ins(f'getelementptr i8, ptr addrspace(3) @global_smem, i32 {128 if tma_load else 16}')
            p = ins(f'getelementptr float, ptr addrspace(3) {smem_out}, i64 {idx64}')
        else:
            p = ins(f'getelementptr float, ptr addrspace(1) {optr}, i64 {idx64}')
        ys = [ins(f'fmul float {value}, {inverse}') for value in exps[k*4:k*4+4]]
        if quantize:
            quantized = []
            for value in ys:
                scaled = ins(f'fmul float {value}, 0x4150000000000000')
                rounded = ins(f'call float asm "cvt.rni.f32.f32 $0, $1;", "=f,f"(float {scaled})')
                quantized.append(ins(f'fmul float {rounded}, 0x3E90000000000000'))
            ys = quantized
        args = ', '.join(f'float {v}' for v in ys)
        if tma_store:
            emit(f'  call void asm sideeffect "@$5 st.shared.v4.b32 [$4], {{$0,$1,$2,$3}};", "f,f,f,f,r,b"({args}, ptr addrspace(3) {p}, i1 {pred})')
        elif pred == 'true':
            emit(f'  call void asm sideeffect "st.global.v4.b32 [$4], {{$0,$1,$2,$3}};", "f,f,f,f,l"({args}, ptr addrspace(1) {p})')
        else:
            emit(f'  call void asm sideeffect "@$5 st.global.v4.b32 [$4], {{$0,$1,$2,$3}};", "f,f,f,f,l,b"({args}, ptr addrspace(1) {p}, i1 {pred})')
    if tma_store:
        emit('  call void @llvm.nvvm.barrier0()')
        emit('  call void asm sideeffect "fence.proxy.async.shared::cta;", ""()')
        size = ins('mul i32 %cols, 4')
        emit('  call void asm sideeffect "{ .reg .pred p; setp.eq.u32 p, $3, 0; '
             '@p cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; '
             '@p cp.async.bulk.commit_group; @p cp.async.bulk.wait_group.read 0; }", '
             f'"l,r,r,r"(ptr addrspace(1) {optr}, ptr addrspace(3) {smem_out}, i32 {size}, i32 {tid})')
    if persistent:
        emit('  br label %advance')
        emit('advance:')
        emit(f'  %nextrow = add i32 {row}, {step}')
        emit('  br label %row_loop')
    else:
        emit('  br label %exit')
    emit('exit:\n  ret void\n}')
    emit('declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()')
    emit('declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()')
    if persistent:
        emit('declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()')
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
        write_ir(block, tail_only=True)
