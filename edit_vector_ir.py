"""Direct LLVM text variants for scalar/two-/four-element memory layouts.

Reuse the instruction editor with a mechanically specialized load/store stanza;
no Triton frontend is involved. Production IR is not overwritten by this helper.
"""
from pathlib import Path
import types


def write_vector_ir(block, threads, width, directory):
    assert width in (1, 2, 4)
    source = Path(__file__).with_name('edit_softmax_ir.py').read_text()
    # This specialization uses only the ordinary one-row path.
    source = source.replace('row_threads * 4', 'row_threads * VECTOR_WIDTH')
    source = source.replace("col = ins(f'mul i32 {row_tid}, 4')", "col = ins(f'mul i32 {row_tid}, {VECTOR_WIDTH}')")
    start = source.index('        if full and (rows_per_cta == 1 or warp_rows):')
    end = source.index('        groups.append((idx64, pred))', start)
    source = source[:start] + '''        dtype = 'float' if VECTOR_WIDTH == 1 else '{' + ', '.join(['float'] * VECTOR_WIDTH) + '}'
        regs = ','.join('$' + str(j) for j in range(VECTOR_WIDTH))
        registers = regs if VECTOR_WIDTH == 1 else '{' + regs + '}'
        opcode = 'ld.global' + ('' if VECTOR_WIDTH == 1 else '.v' + str(VECTOR_WIDTH)) + '.b32'
        constraints = ','.join(['=f'] * VECTOR_WIDTH + ['l'])
        code = f'{opcode} {registers}, [${VECTOR_WIDTH}];'
        args = f'ptr addrspace(1) {p}'
        if pred != 'true':
            init = ' '.join(f'mov.b32 ${j}, 0xff800000;' for j in range(VECTOR_WIDTH))
            code = init + f' @${VECTOR_WIDTH + 1} ' + code
            constraints += ',b'
            args += f', i1 {pred}'
        vec = ins(f'call {dtype} asm sideeffect "{code}", "{constraints}"({args})')
        if VECTOR_WIDTH == 1:
            vals.append(vec)
        else:
            vals.extend(ins(f'extractvalue {dtype} {vec}, {j}') for j in range(VECTOR_WIDTH))
''' + source[end:]
    start = source.index('    for k, (idx64, pred) in enumerate(groups):')
    end = source.index('    if tma_store:\n', source.index('    if tma_store:\n', start) + 1)
    # Replace the entire ordinary output stanza up to the post-store TMA clause.
    end = source.index("    if tma_store:\n        emit('  call void @llvm.nvvm.barrier0()')", start)
    source = source[:start] + '''    for k, (idx64, pred) in enumerate(groups):
        p = ins(f'getelementptr float, ptr addrspace(1) {optr}, i64 {idx64}')
        ys = [ins(f'fmul float {value}, {inverse}') for value in exps[k*VECTOR_WIDTH:(k+1)*VECTOR_WIDTH]]
        regs = ','.join('$' + str(j) for j in range(VECTOR_WIDTH))
        registers = regs if VECTOR_WIDTH == 1 else '{' + regs + '}'
        opcode = 'st.global' + ('' if VECTOR_WIDTH == 1 else '.v' + str(VECTOR_WIDTH)) + '.b32'
        constraints = ','.join(['f'] * VECTOR_WIDTH + ['l'])
        code = f'{opcode} [${VECTOR_WIDTH}], {registers};'
        args = ', '.join(f'float {v}' for v in ys) + f', ptr addrspace(1) {p}'
        if pred != 'true':
            code = f'@${VECTOR_WIDTH + 1} ' + code
            constraints += ',b'
            args += f', i1 {pred}'
        emit(f'  call void asm sideeffect "{code}", "{constraints}"({args})')
''' + source[end:]
    module = types.ModuleType('_vector_editor')
    module.VECTOR_WIDTH = width
    exec(compile(source, 'edit_softmax_ir.py', 'exec'), module.__dict__)
    return module.write_ir(block, threads, warp_rows=False, directory=directory, tail_only=True)
