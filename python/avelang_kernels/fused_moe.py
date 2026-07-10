"""AMDGPU FP8 block-scale fused MoE kernel transcribed from petit-kernel."""

import avelang
import avelang.language as S
import torch


WARP_SIZE = 64
ROW_NEW_BCAST_BASE = 0x150
BF16_PACK_SELECTOR = 0x07060302

GROUP_M = 32
GROUP_N = 256
GROUP_DIM = 256
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS
SCALE_BLOCK_SIZE = 128
TOKEN_BATCH = 8
SUBGROUP_SIZE = 16
VEC_SIZE = 16
ROUTES_PER_BLOCK = TOKEN_BATCH * NUM_WARPS
REF_BUFFER_RANGE = 0xFFFFFFF0
STAGE_COUNT = 2

INPUT_SHM_PADDING_BYTES = 32 * NUM_WARPS
INPUT_SHM_ELEMENTS = TOKEN_BATCH * THREADS + INPUT_SHM_PADDING_BYTES // 4
INPUT_SHM_ELEMENTS_PER_WARP = INPUT_SHM_ELEMENTS // NUM_WARPS
INPUT_SHM_VEC4_PER_WARP = INPUT_SHM_ELEMENTS_PER_WARP // 4
INPUT_STAGE_DWORDS = INPUT_SHM_ELEMENTS + THREADS

SHM_X0_BASE = 0
SHM_SCALE_X0_BASE = SHM_X0_BASE + INPUT_SHM_ELEMENTS
SHM_X1_BASE = INPUT_STAGE_DWORDS
SHM_SCALE_X1_BASE = SHM_X1_BASE + INPUT_SHM_ELEMENTS
SHM_MAX_BASE = STAGE_COUNT * INPUT_STAGE_DWORDS
SHM_Q_H_BASE = SHM_MAX_BASE + THREADS * 2
SHM_RET_BASE = SHM_Q_H_BASE + THREADS * 8
SHM_TOTAL_DWORDS = SHM_RET_BASE + 4 * 0x88 * 8
RET_DWORDS = 4 * 0x88 * 8

W13_TILE_LOADS = 8
W2_TILE_LOADS = 8


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _stage1_scale_dpp_ctrl(stage: int, col: int) -> int:
    return (stage << 1) + (col >> 1)


def _stage2_scale_dpp_ctrl(stage: int, col: int) -> int:
    return (stage << 1) + (col >> 1)


def _stage2_dq_idx(i: int, k: int) -> int:
    return k * 2 + i


@avelang.jit
def _abs_f32(x: S.f32) -> S.f32:
    mask = S.convert(0x7FFFFFFF, S.u32)
    return S.bitcast(S.bitcast(x, S.u32) & mask, S.f32)


@avelang.jit
def _max_f32(lhs: S.f32, rhs: S.f32) -> S.f32:
    if rhs > lhs:
        return rhs
    return lhs


@avelang.jit
def _max_nonneg_f32(lhs: S.f32, rhs: S.f32) -> S.f32:
    lhs_u = S.bitcast(lhs, S.u32)
    rhs_u = S.bitcast(rhs, S.u32)
    return S.bitcast(S.max(lhs_u, rhs_u), S.f32)


@avelang.jit
def _fma4(
    ret: S.Tensor((4,), S.f32),
    a: S.Tensor((4,), S.f32),
    s: S.f32,
    c: S.Tensor((4,), S.f32),
):
    for i in S.range(4):
        ret[i] = a[i] + s * c[i]


@avelang.jit
def _max4(v: S.Tensor((4,), S.f32)) -> S.f32:
    m = _abs_f32(v[0])
    for i in S.range(3):
        av = _abs_f32(v[i + 1])
        if av > m:
            m = av
    return m


@avelang.jit
def _get_dpp_value_ctrl0(src: S.f32) -> S.f32:
    return S.amdgpu.get_dpp(src, src, ROW_NEW_BCAST_BASE + 0, 0xF, 0xF, 0)


@avelang.jit
def _get_dpp_value_ctrl1(src: S.f32) -> S.f32:
    return S.amdgpu.get_dpp(src, src, ROW_NEW_BCAST_BASE + 1, 0xF, 0xF, 0)


@avelang.jit
def _get_dpp_value_ctrl2(src: S.f32) -> S.f32:
    return S.amdgpu.get_dpp(src, src, ROW_NEW_BCAST_BASE + 2, 0xF, 0xF, 0)


@avelang.jit
def _get_dpp_value_ctrl3(src: S.f32) -> S.f32:
    return S.amdgpu.get_dpp(src, src, ROW_NEW_BCAST_BASE + 3, 0xF, 0xF, 0)


@avelang.jit
def _matrix_layout_fetch_scale(
    scales_rsrc: S.Tensor((4,), S.u32),
    stride_n: S.u32,
    tid: S.u32,
    s_offset_bytes: S.u32,
) -> S.f32:
    vindex = ((tid & 1) * (stride_n // SCALE_BLOCK_SIZE)) * 4 + (tid & 2) * 2
    u = S.amdgpu.raw_buffer_load_x1(scales_rsrc, vindex, s_offset_bytes, 0)
    return S.bitcast(u, S.f32)


@avelang.jit
def _w13_load_tile_stage0(
    reg: S.Tensor((W13_TILE_LOADS, 4), S.u32),
    values_rsrc: S.Tensor((4,), S.u32),
    v_offset_bytes: S.u32,
    stride_n: S.u32,
    wid: S.u32,
    wtid: S.u32,
):
    vindex = wtid * 16
    soffset = v_offset_bytes + wid * 16 * stride_n
    row_stride_bytes = WARP_SIZE * stride_n
    for i in S.range(4):
        for j in S.range(2):
            reg[i * 2 + j] = S.amdgpu.raw_buffer_load_x4(
                values_rsrc,
                vindex,
                soffset + i * row_stride_bytes + j * WARP_SIZE * 16,
                0,
            )


@avelang.jit
def _w13_load_tile_stage1(
    reg: S.Tensor((W13_TILE_LOADS, 4), S.u32),
    values_rsrc: S.Tensor((4,), S.u32),
    v_offset_bytes: S.u32,
    stride_n: S.u32,
    wid: S.u32,
    wtid: S.u32,
):
    vindex = wtid * 16
    soffset = v_offset_bytes + wid * 16 * stride_n
    row_stride_bytes = WARP_SIZE * stride_n
    for i in S.range(4):
        for j in S.range(2):
            reg[i * 2 + j] = S.amdgpu.raw_buffer_load_x4(
                values_rsrc,
                vindex,
                soffset + i * row_stride_bytes + (2 + j) * WARP_SIZE * 16,
                0,
            )


@avelang.jit
def _w13_load_scale(
    scales_rsrc: S.Tensor((4,), S.u32),
    stride_n: S.u32,
    tid: S.u32,
    s_offset_bytes: S.u32,
) -> S.f32:
    vindex = ((tid & 1) * (stride_n // SCALE_BLOCK_SIZE)) * 4 + (tid & 2) * 2
    u = S.amdgpu.raw_buffer_load_x1(scales_rsrc, vindex, s_offset_bytes, 0)
    return S.bitcast(u, S.f32)


@avelang.jit
def _w2_load_tile_stage0(
    reg: S.Tensor((W2_TILE_LOADS, 4), S.u32),
    values_rsrc: S.Tensor((4,), S.u32),
    v_offset_bytes: S.u32,
    stride_n: S.u32,
    wid: S.u32,
    wtid: S.u32,
):
    vindex = wtid * 16
    soffset = v_offset_bytes + wid * 16 * stride_n
    inner_step = WARP_SIZE * 16
    for i in S.range(2):
        for j in S.range(4):
            reg[j * 2 + i] = S.amdgpu.raw_buffer_load_x4(
                values_rsrc,
                vindex,
                soffset + i * inner_step + 64 * j * stride_n,
                0,
            )


@avelang.jit
def _w2_load_tile_stage1(
    reg: S.Tensor((W2_TILE_LOADS, 4), S.u32),
    values_rsrc: S.Tensor((4,), S.u32),
    v_offset_bytes: S.u32,
    stride_n: S.u32,
    wid: S.u32,
    wtid: S.u32,
):
    vindex = wtid * 16
    soffset = v_offset_bytes + wid * 16 * stride_n
    inner_step = WARP_SIZE * 16
    for i in S.range(2):
        for j in S.range(4):
            reg[j * 2 + i] = S.amdgpu.raw_buffer_load_x4(
                values_rsrc,
                vindex,
                soffset + (2 + i) * inner_step + 64 * j * stride_n,
                0,
            )


@avelang.jit
def _w2_load_scale(
    scales_rsrc: S.Tensor((4,), S.u32),
    stride_n: S.u32,
    tid: S.u32,
    s_offset_bytes: S.u32,
) -> S.f32:
    return _matrix_layout_fetch_scale(scales_rsrc, stride_n, tid, s_offset_bytes)


@avelang.jit
def _input_fetch_async(
    shared_words: S.Tensor((SHM_TOTAL_DWORDS,), S.u32),
    values_rsrc: S.Tensor((4,), S.u32),
    shm_x_base: S.u32,
    dim: S.u32,
    d: S.u32,
    wid: S.u32,
    wtid: S.u32,
    tokens: S.Tensor((TOKEN_BATCH,), S.u32),
):
    warp_base_bytes = (shm_x_base + wid * INPUT_SHM_ELEMENTS_PER_WARP) * 4
    vindex = wtid * 4
    for i in S.range(TOKEN_BATCH):
        src_off = tokens[i] * dim + d
        lds_off = warp_base_bytes + i * WARP_SIZE * 4
        S.amdgpu.raw_buffer_load_x1_lds(
            values_rsrc,
            shared_words,
            4,
            vindex,
            src_off,
            lds_off,
            0,
        )


@avelang.jit
def _input_fetch_to_regs(
    regs: S.Tensor((8, 4), S.u32),
    shm_x_u128: S.Tensor((INPUT_SHM_ELEMENTS // 4, 4), S.u32),
    wtid: S.u32,
):
    base = INPUT_SHM_VEC4_PER_WARP * (wtid & 3) + ((wtid >> 2) & 3) * SUBGROUP_SIZE + wtid // SUBGROUP_SIZE
    for i in S.range(2):
        for j in S.range(4):
            regs[i * 4 + j] = shm_x_u128[base + i * WARP_SIZE + j * 4]


@avelang.jit
def _input_fetch_scale_async(
    shared_words: S.Tensor((SHM_TOTAL_DWORDS,), S.u32),
    scales_rsrc: S.Tensor((4,), S.u32),
    shm_scale_base: S.u32,
    m: S.u32,
    d: S.u32,
    wid: S.u32,
    wtid: S.u32,
    token_select_x: S.u32,
    token_select_y: S.u32,
):
    token_off = token_select_x * 4
    if (wid & 1) != 0:
        token_off = token_select_y * 4
    scale_block_pair = d // GROUP_DIM
    src_off = (wid // 2) * m * 4 + scale_block_pair * (m * 2 * 4)
    lds_off = (shm_scale_base + wid * WARP_SIZE) * 4
    S.amdgpu.raw_buffer_load_x1_lds(
        scales_rsrc,
        shared_words,
        4,
        token_off,
        src_off,
        lds_off,
        0,
    )


@avelang.jit
def _input_fetch_scale_to_reg(
    ret: S.Tensor((4,), S.f32),
    shm_scale_x: S.Tensor((THREADS,), S.f32),
    tid: S.u32,
):
    for i in S.range(4):
        ret[i] = shm_scale_x[tid + i * WARP_SIZE]


@avelang.jit
def _load_sorted_weights(
    ret: S.Tensor((2,), S.f32),
    sorted_weights_ptr: S.Pointer(S.u32),
    sorted_weights_len: S.u32,
    tid: S.u32,
    route_base: S.u32,
):
    sorted_weights = S.make_tensor(
        sorted_weights_ptr,
        S.u32,
        S.make_layout((sorted_weights_len,), (1,)),
    )
    lane = tid % SUBGROUP_SIZE
    ret[0] = S.bitcast(sorted_weights[route_base + lane], S.f32)
    ret[1] = S.bitcast(sorted_weights[route_base + lane + SUBGROUP_SIZE], S.f32)


@avelang.jit
def _clear_f32x4_matrix(mat: S.Tensor((8, 4), S.f32)):
    zero = S.convert(0.0, S.f32)
    for i in S.range(8):
        for j in S.range(4):
            mat[i, j] = zero


@avelang.jit
def _matmul_stage0(
    t: S.Tensor((8, 4), S.f32),
    w: S.Tensor((8, 4), S.u32),
    x: S.Tensor((8, 4), S.u32),
    x_scale: S.Tensor((4,), S.f32),
    weight_scale: S.f32,
):
    w_sdpp_vals = S.make_local((2,), S.f32)
    w_sdpp_vals[0] = S.amdgpu.get_dpp(weight_scale, weight_scale, ROW_NEW_BCAST_BASE + 0, 0xF, 0xF, 0)
    w_sdpp_vals[1] = S.amdgpu.get_dpp(weight_scale, weight_scale, ROW_NEW_BCAST_BASE + 1, 0xF, 0xF, 0)
    w_u2 = S.view(w, S.Tensor((16, 2), S.u32))
    x_u2 = S.view(x, S.Tensor((16, 2), S.u32))
    for i in S.range(4):
        for row in S.range(2):
            acc = S.make_local((4,), S.f32)
            for c in S.range(4):
                acc[c] = S.convert(0.0, S.f32)
            x_base = row * 8
            for j in S.range(4):
                w_vec = S.view(w_u2[i * 4 + j], S.Tensor((2,), S.u32))
                x_vec = S.view(x_u2[x_base + j], S.Tensor((2,), S.u32))
                acc = S.amdgpu.mfma_f32_16x16x32_fp8_fp8(
                    w_vec,
                    x_vec,
                    acc,
                )
            x_s = x_scale[row]
            s = w_sdpp_vals[i >> 1] * x_s
            scale_pair = S.make_local((2,), S.f32)
            scale_pair[0] = s
            scale_pair[1] = s
            for pair in S.range(2):
                src_pair = S.make_local((2,), S.f32)
                dst_pair = S.make_local((2,), S.f32)
                for elem in S.range(2):
                    src_pair[elem] = acc[pair * 2 + elem]
                    dst_pair[elem] = t[i * 2 + row, pair * 2 + elem]
                pair_val = S.make_local((2,), S.f32)
                for elem in S.range(2):
                    pair_val[elem] = scale_pair[elem] * src_pair[elem] + dst_pair[elem]
                    t[i * 2 + row, pair * 2 + elem] = pair_val[elem]


@avelang.jit
def _matmul_stage1(
    t: S.Tensor((8, 4), S.f32),
    w: S.Tensor((8, 4), S.u32),
    x: S.Tensor((8, 4), S.u32),
    x_scale: S.Tensor((4,), S.f32),
    weight_scale: S.f32,
):
    w_sdpp_vals = S.make_local((2,), S.f32)
    w_sdpp_vals[0] = S.amdgpu.get_dpp(weight_scale, weight_scale, ROW_NEW_BCAST_BASE + 2, 0xF, 0xF, 0)
    w_sdpp_vals[1] = S.amdgpu.get_dpp(weight_scale, weight_scale, ROW_NEW_BCAST_BASE + 3, 0xF, 0xF, 0)
    w_u2 = S.view(w, S.Tensor((16, 2), S.u32))
    x_u2 = S.view(x, S.Tensor((16, 2), S.u32))
    for i in S.range(4):
        for row in S.range(2):
            acc = S.make_local((4,), S.f32)
            for c in S.range(4):
                acc[c] = S.convert(0.0, S.f32)
            x_base = row * 8 + 4
            for j in S.range(4):
                w_vec = S.view(w_u2[i * 4 + j], S.Tensor((2,), S.u32))
                x_vec = S.view(x_u2[x_base + j], S.Tensor((2,), S.u32))
                acc = S.amdgpu.mfma_f32_16x16x32_fp8_fp8(
                    w_vec,
                    x_vec,
                    acc,
                )
            x_s = x_scale[row + 2]
            s = w_sdpp_vals[i >> 1] * x_s
            scale_pair = S.make_local((2,), S.f32)
            scale_pair[0] = s
            scale_pair[1] = s
            for pair in S.range(2):
                src_pair = S.make_local((2,), S.f32)
                dst_pair = S.make_local((2,), S.f32)
                for elem in S.range(2):
                    src_pair[elem] = acc[pair * 2 + elem]
                    dst_pair[elem] = t[i * 2 + row, pair * 2 + elem]
                pair_val = S.make_local((2,), S.f32)
                for elem in S.range(2):
                    pair_val[elem] = scale_pair[elem] * src_pair[elem] + dst_pair[elem]
                    t[i * 2 + row, pair * 2 + elem] = pair_val[elem]


@avelang.jit
def _silu_dot(
    ret: S.Tensor((4,), S.f32),
    gate: S.Tensor((4,), S.f32),
    up: S.Tensor((4,), S.f32),
):
    minus_log2e = S.convert(-1.4426950408889634, S.f32)
    one = S.convert(1.0, S.f32)
    for i in S.range(4):
        inv = S.amdgpu.rcp(one + S.exp2(gate[i] * minus_log2e))
        ret[i] = gate[i] * inv * up[i]


@avelang.jit
def _stage1(
    h: S.Tensor((8, 4), S.f32),
    shared_words: S.Tensor((SHM_TOTAL_DWORDS,), S.u32),
    wid: S.u32,
    wtid: S.u32,
    tid: S.u32,
    token_select_x: S.u32,
    token_select_y: S.u32,
    tokens: S.Tensor((TOKEN_BATCH,), S.u32),
    m: S.u32,
    dim: S.u32,
    act_rsrc: S.Tensor((4,), S.u32),
    act_scale_rsrc: S.Tensor((4,), S.u32),
    w1_rsrc: S.Tensor((4,), S.u32),
    w3_rsrc: S.Tensor((4,), S.u32),
    w1_scale_rsrc: S.Tensor((4,), S.u32),
    w3_scale_rsrc: S.Tensor((4,), S.u32),
    w1_base_value_offset_bytes: S.u32,
    w1_base_scale_offset_bytes: S.u32,
    w3_base_value_offset_bytes: S.u32,
    w3_base_scale_offset_bytes: S.u32,
):
    t_gate = S.make_local((8, 4), S.f32)
    t_up = S.make_local((8, 4), S.f32)
    x0 = S.make_local((8, 4), S.u32)
    x1 = S.make_local((8, 4), S.u32)
    w1_stage0 = S.make_local((8, 4), S.u32)
    w1_stage1 = S.make_local((8, 4), S.u32)
    w3_stage0 = S.make_local((8, 4), S.u32)
    w3_stage1 = S.make_local((8, 4), S.u32)
    scale_x0 = S.make_local((4,), S.f32)
    scale_x1 = S.make_local((4,), S.f32)
    _clear_f32x4_matrix(t_gate)
    _clear_f32x4_matrix(t_up)

    shm_x0_words = S.subview(shared_words, (SHM_X0_BASE,), (INPUT_SHM_ELEMENTS,), (1,))
    shm_x0_u128 = S.view(shm_x0_words, S.Tensor((INPUT_SHM_ELEMENTS // 4, 4), S.u32))
    shm_scale_x0_words = S.subview(shared_words, (SHM_SCALE_X0_BASE,), (THREADS,), (1,))
    shm_scale_x0 = S.view(shm_scale_x0_words, S.f32, S.make_layout((THREADS,), (1,)))
    shm_x1_words = S.subview(shared_words, (SHM_X1_BASE,), (INPUT_SHM_ELEMENTS,), (1,))
    shm_x1_u128 = S.view(shm_x1_words, S.Tensor((INPUT_SHM_ELEMENTS // 4, 4), S.u32))
    shm_scale_x1_words = S.subview(shared_words, (SHM_SCALE_X1_BASE,), (THREADS,), (1,))
    shm_scale_x1 = S.view(shm_scale_x1_words, S.f32, S.make_layout((THREADS,), (1,)))

    w1_value_offset_bytes = w1_base_value_offset_bytes
    w1_scale_offset_bytes = w1_base_scale_offset_bytes
    w3_value_offset_bytes = w3_base_value_offset_bytes
    w3_scale_offset_bytes = w3_base_scale_offset_bytes

    _input_fetch_async(shared_words, act_rsrc, SHM_X0_BASE, dim, 0, wid, wtid, tokens)
    _input_fetch_scale_async(
        shared_words,
        act_scale_rsrc,
        SHM_SCALE_X0_BASE,
        m,
        0,
        wid,
        wtid,
        token_select_x,
        token_select_y,
    )
    _w13_load_tile_stage0(w1_stage0, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
    _w13_load_tile_stage1(w1_stage1, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
    scale_w1 = _w13_load_scale(w1_scale_rsrc, dim, tid, w1_scale_offset_bytes)
    w1_value_offset_bytes = w1_value_offset_bytes + 4 * WARP_SIZE * 16
    w1_scale_offset_bytes = w1_scale_offset_bytes + 8
    S.amdgpu.s_waitcnt(0, -1, -1)
    S.syncthreads()
    _input_fetch_to_regs(x0, shm_x0_u128, wtid)
    _input_fetch_scale_to_reg(scale_x0, shm_scale_x0, wtid)

    stage1_iters = (dim + 2 * GROUP_DIM - 1) // (2 * GROUP_DIM)
    for iter_idx in S.range(stage1_iters):
        d = iter_idx * (2 * GROUP_DIM)
        S.syncthreads()

        _input_fetch_async(shared_words, act_rsrc, SHM_X1_BASE, dim, d + GROUP_DIM, wid, wtid, tokens)
        _input_fetch_scale_async(
            shared_words,
            act_scale_rsrc,
            SHM_SCALE_X1_BASE,
            m,
            d + GROUP_DIM,
            wid,
            wtid,
            token_select_x,
            token_select_y,
        )
        _w13_load_tile_stage0(w3_stage0, w3_rsrc, w3_value_offset_bytes, dim, wid, wtid)
        _w13_load_tile_stage1(w3_stage1, w3_rsrc, w3_value_offset_bytes, dim, wid, wtid)
        scale_w3 = _w13_load_scale(w3_scale_rsrc, dim, tid, w3_scale_offset_bytes)
        w3_value_offset_bytes = w3_value_offset_bytes + 4 * WARP_SIZE * 16
        w3_scale_offset_bytes = w3_scale_offset_bytes + 8
        _matmul_stage0(t_gate, w1_stage0, x0, scale_x0, scale_w1)
        _matmul_stage1(t_gate, w1_stage1, x0, scale_x0, scale_w1)
        S.amdgpu.s_waitcnt(0, -1, -1)
        _w13_load_tile_stage0(w1_stage0, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
        _w13_load_tile_stage1(w1_stage1, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
        scale_w1 = _w13_load_scale(w1_scale_rsrc, dim, tid, w1_scale_offset_bytes)
        w1_value_offset_bytes = w1_value_offset_bytes + 4 * WARP_SIZE * 16
        w1_scale_offset_bytes = w1_scale_offset_bytes + 8
        _matmul_stage0(t_up, w3_stage0, x0, scale_x0, scale_w3)
        _matmul_stage1(t_up, w3_stage1, x0, scale_x0, scale_w3)

        if d + GROUP_DIM >= dim:
            break

        S.amdgpu.s_waitcnt(0, -1, -1)
        S.syncthreads()
        _input_fetch_to_regs(x1, shm_x1_u128, wtid)
        _input_fetch_scale_to_reg(scale_x1, shm_scale_x1, wtid)

        _input_fetch_async(shared_words, act_rsrc, SHM_X0_BASE, dim, d + 2 * GROUP_DIM, wid, wtid, tokens)
        _input_fetch_scale_async(
            shared_words,
            act_scale_rsrc,
            SHM_SCALE_X0_BASE,
            m,
            d + 2 * GROUP_DIM,
            wid,
            wtid,
            token_select_x,
            token_select_y,
        )
        _w13_load_tile_stage0(w3_stage0, w3_rsrc, w3_value_offset_bytes, dim, wid, wtid)
        _w13_load_tile_stage1(w3_stage1, w3_rsrc, w3_value_offset_bytes, dim, wid, wtid)
        scale_w3 = _w13_load_scale(w3_scale_rsrc, dim, tid, w3_scale_offset_bytes)
        w3_value_offset_bytes = w3_value_offset_bytes + 4 * WARP_SIZE * 16
        w3_scale_offset_bytes = w3_scale_offset_bytes + 8
        _matmul_stage0(t_gate, w1_stage0, x1, scale_x1, scale_w1)
        _matmul_stage1(t_gate, w1_stage1, x1, scale_x1, scale_w1)
        S.amdgpu.s_waitcnt(0, -1, -1)
        _w13_load_tile_stage0(w1_stage0, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
        _w13_load_tile_stage1(w1_stage1, w1_rsrc, w1_value_offset_bytes, dim, wid, wtid)
        scale_w1 = _w13_load_scale(w1_scale_rsrc, dim, tid, w1_scale_offset_bytes)
        w1_value_offset_bytes = w1_value_offset_bytes + 4 * WARP_SIZE * 16
        w1_scale_offset_bytes = w1_scale_offset_bytes + 8
        _matmul_stage0(t_up, w3_stage0, x1, scale_x1, scale_w3)
        _matmul_stage1(t_up, w3_stage1, x1, scale_x1, scale_w3)

        S.amdgpu.s_waitcnt(0, -1, -1)
        S.syncthreads()
        _input_fetch_to_regs(x0, shm_x0_u128, wtid)
        _input_fetch_scale_to_reg(scale_x0, shm_scale_x0, wtid)

    for i in S.range(8):
        tmp = S.make_local((4,), S.f32)
        _silu_dot(tmp, t_gate[i], t_up[i])
        h[i] = tmp


@avelang.jit
def _compute_row_max(
    shm_max: S.Tensor((THREADS, 2), S.f32),
    h: S.Tensor((8, 4), S.f32),
    tid: S.u32,
    quant_scale: S.Tensor((4,), S.f32),
    dequant_scale: S.Tensor((4,), S.f32),
):
    local_max_floor = S.convert(1.0e-6, S.f32)
    fp8_max = S.convert(240.0, S.f32)
    shm_max_flat = S.view(shm_max, S.Tensor((THREADS * 2,), S.f32))
    for c in S.range(2):
        base = c * 4
        lm0 = _abs_f32(h[base + 0, 0])
        lm2 = _abs_f32(h[base + 2, 0])
        lm1 = _abs_f32(h[base + 1, 0])
        lm3 = _abs_f32(h[base + 3, 0])
        for j in S.range(3):
            v0 = _abs_f32(h[base + 0, j + 1])
            if v0 > lm0:
                lm0 = v0
            v2 = _abs_f32(h[base + 2, j + 1])
            if v2 > lm2:
                lm2 = v2
            v1 = _abs_f32(h[base + 1, j + 1])
            if v1 > lm1:
                lm1 = v1
            v3 = _abs_f32(h[base + 3, j + 1])
            if v3 > lm3:
                lm3 = v3
        lm_x = _max_nonneg_f32(local_max_floor, _max_nonneg_f32(lm0, lm2))
        lm_y = _max_nonneg_f32(local_max_floor, _max_nonneg_f32(lm1, lm3))
        shm_max_flat[tid * 2] = lm_x
        shm_max_flat[tid * 2 + 1] = lm_y
        S.syncthreads()

        lane = tid % SUBGROUP_SIZE
        for i in S.range(16):
            base_idx = (lane + 16 * i) * 2
            v_x = shm_max_flat[base_idx]
            v_y = shm_max_flat[base_idx + 1]
            lm_x = _max_nonneg_f32(lm_x, v_x)
            lm_y = _max_nonneg_f32(lm_y, v_y)

        quant_scale[c * 2 + 0] = fp8_max * S.amdgpu.rcp(lm_x)
        quant_scale[c * 2 + 1] = fp8_max * S.amdgpu.rcp(lm_y)
        dequant_scale[c * 2 + 0] = S.amdgpu.rcp(quant_scale[c * 2 + 0])
        dequant_scale[c * 2 + 1] = S.amdgpu.rcp(quant_scale[c * 2 + 1])
        S.syncthreads()


@avelang.jit
def _quantize_and_shuffle(
    out: S.Tensor((8, 4), S.u32),
    dq_act: S.Tensor((4,), S.f32),
    shm_max: S.Tensor((THREADS, 2), S.f32),
    shm_q_h: S.Tensor((THREADS * 8,), S.u32),
    tid: S.u32,
    wid: S.u32,
    wtid: S.u32,
    h: S.Tensor((8, 4), S.f32),
):
    quant_scale = S.make_local((4,), S.f32)
    _compute_row_max(shm_max, h, tid, quant_scale, dq_act)

    q = S.make_local((8,), S.u32)
    zero = S.convert(0, S.u32)
    for i in S.range(8):
        row = i % 2
        half = i // 4
        s = quant_scale[half * 2 + row]
        v = h[i]
        u0 = S.amdgpu.cvt_pk_fp8_f32(v[0] * s, v[1] * s, zero, 0)
        u1 = S.amdgpu.cvt_pk_fp8_f32(v[2] * s, v[3] * s, zero, 0)
        q[i] = (u1 << 16) | u0

    for i in S.range(8):
        idx = (
            wid * WARP_SIZE
            + (wtid // 32) * 32
            + (wtid % SUBGROUP_SIZE) * 2
            + (wtid % 32) // SUBGROUP_SIZE
            + i // 2 * THREADS
            + (i % 2) * 1024
        )
        shm_q_h[idx] = q[i]
    S.syncthreads()

    shm_q_h_u2 = S.view(shm_q_h, S.Tensor((THREADS * 8 // 2, 2), S.u32))
    out_u2 = S.view(out, S.Tensor((16, 2), S.u32))
    for i in S.range(8):
        base = (wtid // SUBGROUP_SIZE) * 32 + (wtid % SUBGROUP_SIZE)
        idx0 = base + i * 128
        out_u2[i * 2 + 0] = shm_q_h_u2[idx0]
        out_u2[i * 2 + 1] = shm_q_h_u2[idx0 + 16]
    S.syncthreads()


@avelang.jit
def _multiply_route_weights(
    t: S.Tensor((8, 4), S.f32),
    sorted_weights: S.Tensor((2,), S.f32),
):
    for i in S.range(2):
        rw = sorted_weights[i]
        for j in S.range(4):
            for k in S.range(4):
                t[j * 2 + i, k] = t[j * 2 + i, k] * rw


@avelang.jit
def _to_bf16_rn(
    ret: S.Tensor((2,), S.u32),
    m: S.Tensor((4,), S.f32),
):
    m_bits = S.make_local((4,), S.u32)
    round_bias = S.convert(0x8000, S.u32)
    nan_bits = S.convert(0x7FFF0000, S.u32)
    for j in S.range(4):
        f = m[j]
        u = S.bitcast(f, S.u32)
        rounded = u + round_bias
        if f != f:
            m_bits[j] = nan_bits
        else:
            m_bits[j] = rounded
    selector = S.convert(BF16_PACK_SELECTOR, S.u32)
    ret[0] = S.amdgpu.perm(m_bits[1], m_bits[0], selector)
    ret[1] = S.amdgpu.perm(m_bits[3], m_bits[2], selector)


@avelang.jit
def _stage2_write_back(
    out_bf16: S.Tensor((2,), S.bf16),
    shm_ret: S.Tensor((RET_DWORDS,), S.u32),
    tokens: S.Tensor((TOKEN_BATCH,), S.u32),
    num_tokens: S.u32,
    dim: S.u32,
    d: S.u32,
    wid: S.u32,
    wtid: S.u32,
):
    idx_r = (wtid >> 1) * 34 + (wtid & 1) + wid * 2
    for i in S.range(8):
        read_off_dw = (i // 4) * 2176 + (i % 4) * 8
        o0 = shm_ret[idx_r + read_off_dw]
        o1 = shm_ret[idx_r + read_off_dw + 1088]
        if tokens[i] < num_tokens:
            voffset_bytes = (tokens[i] * dim + d + wtid * 2) * 2
            o0_words = S.make_local((1,), S.u32)
            o1_words = S.make_local((1,), S.u32)
            o0_words[0] = o0
            o1_words[0] = o1
            o0_bf16 = S.view(o0_words, S.Tensor((1, 2), S.bf16))[0]
            o1_bf16 = S.view(o1_words, S.Tensor((1, 2), S.bf16))[0]
            S.amdgpu.atomic_add(
                voffset_bytes, o0_bf16, out_bf16, 1
            )
            S.amdgpu.atomic_add(
                voffset_bytes + 256, o1_bf16, out_bf16, 1
            )


@avelang.jit
def _stage2(
    out_bf16: S.Tensor((2,), S.bf16),
    shm_ret: S.Tensor((RET_DWORDS,), S.u32),
    quant_h: S.Tensor((8, 4), S.u32),
    dq_act: S.Tensor((4,), S.f32),
    sorted_weights: S.Tensor((2,), S.f32),
    tokens: S.Tensor((TOKEN_BATCH,), S.u32),
    num_tokens: S.u32,
    tid: S.u32,
    wid: S.u32,
    wtid: S.u32,
    dim: S.u32,
    inter_dim: S.u32,
    w2_rsrc: S.Tensor((4,), S.u32),
    w2_scale_rsrc: S.Tensor((4,), S.u32),
    w2_base_value_offset_bytes: S.u32,
    w2_base_scale_offset_bytes: S.u32,
):
    w2_stage0 = S.make_local((8, 4), S.u32)
    w2_stage1 = S.make_local((8, 4), S.u32)
    w2_value_offset_bytes = w2_base_value_offset_bytes
    w2_scale_offset_bytes = w2_base_scale_offset_bytes

    for d in S.range(dim // GROUP_N):
        t = S.make_local((8, 4), S.f32)
        _clear_f32x4_matrix(t)

        _w2_load_tile_stage0(w2_stage0, w2_rsrc, w2_value_offset_bytes, inter_dim, wid, wtid)
        _w2_load_tile_stage1(w2_stage1, w2_rsrc, w2_value_offset_bytes, inter_dim, wid, wtid)
        scale_w2 = _w2_load_scale(w2_scale_rsrc, inter_dim, tid, w2_scale_offset_bytes)
        _matmul_stage0(t, w2_stage0, quant_h, dq_act, scale_w2)
        _matmul_stage1(t, w2_stage1, quant_h, dq_act, scale_w2)

        _multiply_route_weights(t, sorted_weights)

        for i in S.range(8):
            o = S.make_local((2,), S.u32)
            _to_bf16_rn(o, t[i])
            col = i // 2
            row = i % 2
            idx_w = (wtid >> 4) * 34 + (wtid & 15) * 2 + wid * 136
            write_off_dw = col * 544 + row * 2176
            shm_ret[idx_w + write_off_dw] = o[0]
            shm_ret[idx_w + write_off_dw + 1] = o[1]
        S.syncthreads()

        _stage2_write_back(out_bf16, shm_ret, tokens, num_tokens, dim, d * GROUP_N, wid, wtid)
        S.syncthreads()

        w2_value_offset_bytes = w2_value_offset_bytes + inter_dim * GROUP_N
        w2_scale_offset_bytes = w2_scale_offset_bytes + (inter_dim // SCALE_BLOCK_SIZE) * 8


@avelang.jit
def _fused_moe_blockscale_fp8_kernel(
    out_ptr: S.Pointer(S.bf16),
    act_ptr: S.Pointer(S.u8),
    w13_ptr: S.Pointer(S.u8),
    w2_ptr: S.Pointer(S.u8),
    sorted_token_ids_ptr: S.Pointer(S.u32),
    sorted_weights_ptr: S.Pointer(S.u32),
    sorted_expert_ids_ptr: S.Pointer(S.u32),
    num_valid_ids_ptr: S.Pointer(S.u32),
    topk: S.u32,
    scales_act_ptr: S.Pointer(S.u32),
    scales_w13_ptr: S.Pointer(S.u32),
    scales_w2_ptr: S.Pointer(S.u32),
    m: S.u32,
    dim: S.u32,
    inter_dim: S.u32,
):
    del topk

    n_blocks = dim // SCALE_BLOCK_SIZE
    k_blocks = inter_dim // SCALE_BLOCK_SIZE
    tid = S.thread_id(0)
    tile_k = S.block_id(0)
    route_group = S.block_id(1)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    col_id = tid % SUBGROUP_SIZE

    g_num_valid_ids = S.make_tensor(num_valid_ids_ptr, S.u32, S.make_layout((2,), (1,)))
    num_valid_ids = g_num_valid_ids[0]
    num_tokens = g_num_valid_ids[1]
    num_valid_m_blocks = (num_valid_ids + ROUTES_PER_BLOCK - 1) // ROUTES_PER_BLOCK
    route_base = route_group * ROUTES_PER_BLOCK

    if route_group >= num_valid_m_blocks:
        return
    if route_base >= num_valid_ids:
        return

    g_sorted_token_ids = S.make_tensor(sorted_token_ids_ptr, S.u32, S.make_layout((num_valid_ids,), (1,)))
    g_sorted_expert_ids = S.make_tensor(sorted_expert_ids_ptr, S.u32, S.make_layout((num_valid_m_blocks,), (1,)))
    expert_id = g_sorted_expert_ids[route_group]

    act_memref = S.make_tensor(act_ptr, S.u8, S.make_layout((m * dim,), (1,)))
    w13_memref = S.make_tensor(w13_ptr, S.u8, S.make_layout((REF_BUFFER_RANGE,), (1,)))
    w2_memref = S.make_tensor(w2_ptr, S.u8, S.make_layout((REF_BUFFER_RANGE,), (1,)))
    scale_act_memref = S.make_tensor(scales_act_ptr, S.u32, S.make_layout((n_blocks * m,), (1,)))
    scale_w13_memref = S.make_tensor(scales_w13_ptr, S.u32, S.make_layout((REF_BUFFER_RANGE // 4,), (1,)))
    scale_w2_memref = S.make_tensor(scales_w2_ptr, S.u32, S.make_layout((REF_BUFFER_RANGE // 4,), (1,)))
    out_bf16 = S.make_tensor(out_ptr, S.bf16, S.make_layout((2,), (1,)))

    act_rsrc = S.amdgpu.make_rsrc(act_memref, m * dim)
    scale_act_rsrc = S.amdgpu.make_rsrc(scale_act_memref, n_blocks * m * 4)
    w13_rsrc_all = S.amdgpu.make_rsrc(w13_memref, REF_BUFFER_RANGE)
    w2_rsrc_all = S.amdgpu.make_rsrc(w2_memref, REF_BUFFER_RANGE)
    scale_w13_rsrc_all = S.amdgpu.make_rsrc(scale_w13_memref, REF_BUFFER_RANGE)
    scale_w2_rsrc_all = S.amdgpu.make_rsrc(scale_w2_memref, REF_BUFFER_RANGE)

    token_select_x = g_sorted_token_ids[route_base + col_id] & 0x00FFFFFF
    token_select_y = g_sorted_token_ids[route_base + col_id + SUBGROUP_SIZE] & 0x00FFFFFF

    tokens = S.make_local((TOKEN_BATCH,), S.u32)
    for i in S.range(TOKEN_BATCH):
        token_idx = wid + i * 4
        tokens[i] = g_sorted_token_ids[route_base + token_idx] & 0x00FFFFFF

    sorted_weights = S.make_local((2,), S.f32)
    _load_sorted_weights(sorted_weights, sorted_weights_ptr, num_valid_ids, tid, route_base)

    shared_words = S.make_shared((SHM_TOTAL_DWORDS,), S.u32)
    shm_max_words = S.subview(shared_words, (SHM_MAX_BASE,), (THREADS * 2,), (1,))
    shm_max = S.view(shm_max_words, S.f32, S.make_layout((THREADS, 2), (2, 1)))
    shm_q_h = S.subview(shared_words, (SHM_Q_H_BASE,), (THREADS * 8,), (1,))
    shm_ret = S.subview(shared_words, (SHM_RET_BASE,), (RET_DWORDS,), (1,))

    w1_value_offset_bytes = expert_id * (2 * inter_dim * dim) + tile_k * GROUP_DIM * dim
    w1_scale_offset_bytes = (expert_id * (2 * k_blocks * n_blocks) + tile_k * (GROUP_DIM // SCALE_BLOCK_SIZE) * n_blocks) * 4
    w3_value_offset_bytes = w1_value_offset_bytes + inter_dim * dim
    w3_scale_offset_bytes = w1_scale_offset_bytes + k_blocks * n_blocks * 4
    w2_value_offset_bytes = expert_id * (dim * inter_dim) + tile_k * (GROUP_DIM * 16)
    w2_scale_offset_bytes = (expert_id * (n_blocks * k_blocks) + tile_k * (GROUP_DIM // SCALE_BLOCK_SIZE)) * 4

    w1_rsrc = w13_rsrc_all
    w3_rsrc = w13_rsrc_all
    w2_rsrc = w2_rsrc_all
    w1_scale_rsrc = scale_w13_rsrc_all
    w3_scale_rsrc = scale_w13_rsrc_all
    w2_scale_rsrc = scale_w2_rsrc_all
    act_scale_rsrc = scale_act_rsrc

    h = S.make_local((8, 4), S.f32)
    _stage1(
        h,
        shared_words,
        wid,
        wtid,
        tid,
        token_select_x,
        token_select_y,
        tokens,
        m,
        dim,
        act_rsrc,
        act_scale_rsrc,
        w1_rsrc,
        w3_rsrc,
        w1_scale_rsrc,
        w3_scale_rsrc,
        w1_value_offset_bytes,
        w1_scale_offset_bytes,
        w3_value_offset_bytes,
        w3_scale_offset_bytes,
    )
    S.syncthreads()

    quant_h = S.make_local((8, 4), S.u32)
    dq_act = S.make_local((4,), S.f32)
    _quantize_and_shuffle(quant_h, dq_act, shm_max, shm_q_h, tid, wid, wtid, h)

    _stage2(
        out_bf16,
        shm_ret,
        quant_h,
        dq_act,
        sorted_weights,
        tokens,
        num_tokens,
        tid,
        wid,
        wtid,
        dim,
        inter_dim,
        w2_rsrc,
        w2_scale_rsrc,
        w2_value_offset_bytes,
        w2_scale_offset_bytes,
    )


def _validate_fused_moe_fp8_blockscale_inputs(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
):
    fp8_dtypes = {torch.uint8}
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            fp8_dtypes.add(dtype)

    if input_q.ndim != 2:
        raise ValueError(f"input_q must be rank-2 (got {input_q.ndim})")
    if w13_q.ndim != 3 or w2_q.ndim != 3:
        raise ValueError("w13_q and w2_q must be rank-3")
    if input_q.dtype not in fp8_dtypes or w13_q.dtype not in fp8_dtypes or w2_q.dtype not in fp8_dtypes:
        raise TypeError(
            "input_q, w13_q, and w2_q must be uint8, float8_e4m3fn, or "
            f"float8_e4m3fnuz (got input_q={input_q.dtype}, w13_q={w13_q.dtype}, w2_q={w2_q.dtype})"
        )
    if input_q.device.type != "cuda" or w13_q.device != input_q.device or w2_q.device != input_q.device:
        raise ValueError("all quantized tensors must be on the same CUDA device")
    if not input_q.is_contiguous() or not w13_q.is_contiguous() or not w2_q.is_contiguous():
        raise ValueError("input_q, w13_q, and w2_q must be contiguous")

    tokens, dim = input_q.shape
    experts, w13_rows, w13_cols = w13_q.shape
    w2_experts, w2_rows, inter_dim = w2_q.shape
    if w13_cols != dim:
        raise ValueError(f"w13_q.shape[2] must equal model dim {dim} (got {w13_cols})")
    if w2_experts != experts or w2_rows != dim or w13_rows != 2 * inter_dim:
        raise ValueError(
            "weight shapes must match petit-kernel layout: "
            f"w13_q={tuple(w13_q.shape)}, w2_q={tuple(w2_q.shape)}, input_q={tuple(input_q.shape)}"
        )
    if dim == 0 or inter_dim == 0 or dim % SCALE_BLOCK_SIZE != 0 or inter_dim % SCALE_BLOCK_SIZE != 0:
        raise ValueError(f"dim and inter_dim must be nonzero multiples of {SCALE_BLOCK_SIZE}")

    if not isinstance(topk, int) or topk <= 0:
        raise ValueError(f"topk must be a positive int (got {topk})")

    if sorted_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"sorted_token_ids must be int32/int64 (got {sorted_token_ids.dtype})")
    if sorted_weights.dtype != torch.float32:
        raise TypeError(f"sorted_weights must be float32 (got {sorted_weights.dtype})")
    if sorted_expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"sorted_expert_ids must be int32/int64 (got {sorted_expert_ids.dtype})")
    if num_valid_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"num_valid_ids must be int32/int64 (got {num_valid_ids.dtype})")
    if input_scale.dtype != torch.float32 or fc1_scale.dtype != torch.float32 or fc2_scale.dtype != torch.float32:
        raise TypeError("input_scale, fc1_scale, and fc2_scale must be float32")

    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("sorted_weights", sorted_weights),
        ("sorted_expert_ids", sorted_expert_ids),
        ("num_valid_ids", num_valid_ids),
        ("input_scale", input_scale),
        ("fc1_scale", fc1_scale),
        ("fc2_scale", fc2_scale),
    ):
        if tensor.device != input_q.device:
            raise ValueError(f"{name} must be on {input_q.device} (got {tensor.device})")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    n_blocks = dim // SCALE_BLOCK_SIZE
    k_blocks = inter_dim // SCALE_BLOCK_SIZE
    if input_scale.shape != (tokens, n_blocks):
        raise ValueError(
            f"input_scale must have shape {(tokens, n_blocks)} "
            f"(got {tuple(input_scale.shape)})"
        )
    if fc1_scale.shape != (experts, 2 * k_blocks * n_blocks):
        raise ValueError(
            f"fc1_scale must have shape {(experts, 2 * k_blocks * n_blocks)} "
            f"(got {tuple(fc1_scale.shape)})"
        )
    if fc2_scale.shape != (experts, n_blocks * k_blocks):
        raise ValueError(
            f"fc2_scale must have shape {(experts, n_blocks * k_blocks)} "
            f"(got {tuple(fc2_scale.shape)})"
        )
    if num_valid_ids.numel() != 2:
        raise ValueError("num_valid_ids must contain [num_valid_ids, token_count]")


def _fused_moe_fp8_blockscale_g1u1_impl(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_fused_moe_fp8_blockscale_inputs(
        input_q,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale,
        fc1_scale,
        fc2_scale,
    )

    tokens, dim = input_q.shape
    inter_dim = w2_q.shape[-1]
    if out is None:
        out = torch.zeros((tokens, dim), dtype=torch.bfloat16, device=input_q.device)
    else:
        if out.device != input_q.device:
            raise ValueError("out must be on the same device as input_q")
        if out.dtype != torch.bfloat16:
            raise ValueError("out must be bfloat16")
        if out.shape != (tokens, dim):
            raise ValueError("out must be [tokens, dim]")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        out.zero_()

    # The valid route count is device data and must not be synchronously read
    # on the host: doing so breaks CUDA graph capture.  The kernel guards each
    # speculative route group using num_valid_ids[0].
    route_groups = max(1, sorted_expert_ids.numel())
    grid = (_ceil_div(inter_dim, GROUP_DIM), route_groups, 1)
    block = (THREADS, 1, 1)

    sorted_token_ids_i32 = sorted_token_ids.to(dtype=torch.int32)
    sorted_expert_ids_i32 = sorted_expert_ids.to(dtype=torch.int32)
    num_valid_ids_i32 = num_valid_ids.to(dtype=torch.int32)
    sorted_weights_u32 = sorted_weights.view(torch.uint32)
    # The raw-buffer kernel indexes activation scales as [dim_block, token],
    # whereas the public petit-compatible API exposes [token, dim_block].
    input_scale_u32 = input_scale.transpose(0, 1).contiguous().view(torch.uint32)
    fc1_scale_u32 = fc1_scale.view(torch.uint32)
    fc2_scale_u32 = fc2_scale.view(torch.uint32)

    _fused_moe_blockscale_fp8_kernel[lambda: (grid, block)](
        out,
        input_q.view(torch.uint8),
        w13_q.view(torch.uint8),
        w2_q.view(torch.uint8),
        sorted_token_ids_i32,
        sorted_weights_u32,
        sorted_expert_ids_i32,
        num_valid_ids_i32,
        topk,
        input_scale_u32,
        fc1_scale_u32,
        fc2_scale_u32,
        tokens,
        dim,
        inter_dim,
        num_warps=NUM_WARPS,
    )
    return out


def fused_moe_fp8_blockscale_g1u1(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _fused_moe_fp8_blockscale_g1u1_impl(
        input_q,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale,
        fc1_scale,
        fc2_scale,
        out,
    )


__all__ = ["fused_moe_fp8_blockscale_g1u1"]
