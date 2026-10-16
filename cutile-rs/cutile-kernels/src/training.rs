/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Training backward, cross-entropy loss, and optimizer cuTile kernels (f16 / bf16 / f32).

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rms_norm_bwd_dx_f16_module {
    use cutile::core::*;

    /// Computes RMSNorm input gradient `dx` (f16) over the last dimension [tokens, N].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn rms_norm_bwd_dx_f16<const N: i32, const BLOCK_SIZE: i32>(
        dy: &Tensor<f16, { [-1, N] }>,
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        dx: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let dy_part: Partition<f16, { [1, BLOCK_SIZE] }> = dy.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);

        let mut sum_x2: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            sum_x2 = sum_x2 + tx * tx;
            sum_dy_w_x = sum_dy_w_x + tdy * tw * tx;
        }
        let n: f32 = convert_scalar(N);
        let sum_x2_1d: Tile<f32, { [1] }> = reduce_sum(sum_x2, 1i32);
        let sum_dot_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w_x, 1i32);
        let sum_x2_s: f32 = tile_to_scalar(sum_x2_1d.reshape(const_shape![]));
        let sq_mean: f32 = sum_x2_s / n;
        let inv_rms_t: Tile<f32, { [] }> = rsqrt(scalar_to_tile(sq_mean + eps), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms_t);
        let dot_s: f32 = tile_to_scalar(sum_dot_1d.reshape(const_shape![]));
        let c_s: f32 = (dot_s * inv_rms * inv_rms) / n;

        let inv_rms_v: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);
        let c_v: Tile<f32, { [1, BLOCK_SIZE] }> = c_s.broadcast(tile_shape);
        let mut dx_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> = dx.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let tdx: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms_v * (tdy * tw - tx * c_v);
            dx_part.store(convert_tile(tdx), [0i32, j]);
        }
    }
}

pub use rms_norm_bwd_dx_f16_module::rms_norm_bwd_dx_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rms_norm_bwd_dx_bf16_module {
    use cutile::core::*;

    /// Computes RMSNorm input gradient `dx` (bf16) over the last dimension [tokens, N].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn rms_norm_bwd_dx_bf16<const N: i32, const BLOCK_SIZE: i32>(
        dy: &Tensor<bf16, { [-1, N] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        dx: &mut Tensor<bf16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let dy_part: Partition<bf16, { [1, BLOCK_SIZE] }> = dy.partition(tile_shape);
        let x_part: Partition<bf16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let w_part: Partition<bf16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);

        let mut sum_x2: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            sum_x2 = sum_x2 + tx * tx;
            sum_dy_w_x = sum_dy_w_x + tdy * tw * tx;
        }
        let n: f32 = convert_scalar(N);
        let sum_x2_1d: Tile<f32, { [1] }> = reduce_sum(sum_x2, 1i32);
        let sum_dot_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w_x, 1i32);
        let sum_x2_s: f32 = tile_to_scalar(sum_x2_1d.reshape(const_shape![]));
        let sq_mean: f32 = sum_x2_s / n;
        let inv_rms_t: Tile<f32, { [] }> = rsqrt(scalar_to_tile(sq_mean + eps), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms_t);
        let dot_s: f32 = tile_to_scalar(sum_dot_1d.reshape(const_shape![]));
        let c_s: f32 = (dot_s * inv_rms * inv_rms) / n;

        let inv_rms_v: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);
        let c_v: Tile<f32, { [1, BLOCK_SIZE] }> = c_s.broadcast(tile_shape);
        let mut dx_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> = dx.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let tdx: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms_v * (tdy * tw - tx * c_v);
            dx_part.store(convert_tile(tdx), [0i32, j]);
        }
    }
}

pub use rms_norm_bwd_dx_bf16_module::rms_norm_bwd_dx_bf16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod layer_norm_bwd_dx_f16_module {
    use cutile::core::*;

    /// Computes LayerNorm input gradient `dx` (f16) over the last dimension [tokens, N].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn layer_norm_bwd_dx_f16<const N: i32, const BLOCK_SIZE: i32>(
        dy: &Tensor<f16, { [-1, N] }>,
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        dx: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let dy_part: Partition<f16, { [1, BLOCK_SIZE] }> = dy.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);

        let mut sum_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_x2: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let dy_w: Tile<f32, { [1, BLOCK_SIZE] }> = tdy * tw;
            sum_x = sum_x + tx;
            sum_x2 = sum_x2 + tx * tx;
            sum_dy_w = sum_dy_w + dy_w;
            sum_dy_w_x = sum_dy_w_x + dy_w * tx;
        }
        let n: f32 = convert_scalar(N);
        let sum_x_1d: Tile<f32, { [1] }> = reduce_sum(sum_x, 1i32);
        let sum_x2_1d: Tile<f32, { [1] }> = reduce_sum(sum_x2, 1i32);
        let sum_dy_w_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w, 1i32);
        let sum_dy_w_x_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w_x, 1i32);

        let sum_x_s: f32 = tile_to_scalar(sum_x_1d.reshape(const_shape![]));
        let sum_x2_s: f32 = tile_to_scalar(sum_x2_1d.reshape(const_shape![]));
        let sum_dy_w_s: f32 = tile_to_scalar(sum_dy_w_1d.reshape(const_shape![]));
        let sum_dy_w_x_s: f32 = tile_to_scalar(sum_dy_w_x_1d.reshape(const_shape![]));
        let mean: f32 = sum_x_s / n;
        let ex2: f32 = sum_x2_s / n;
        let var: f32 = ex2 - mean * mean;
        let inv_std_t: Tile<f32, { [] }> = rsqrt(scalar_to_tile(var + eps), ftz::Disabled);
        let inv_std: f32 = tile_to_scalar(inv_std_t);

        let c1: f32 = sum_dy_w_s / n;
        let dot_x: f32 = sum_dy_w_x_s / n;
        let c2: f32 = (dot_x - mean * c1) * inv_std;

        let mean_v: Tile<f32, { [1, BLOCK_SIZE] }> = mean.broadcast(tile_shape);
        let inv_std_v: Tile<f32, { [1, BLOCK_SIZE] }> = inv_std.broadcast(tile_shape);
        let c1_v: Tile<f32, { [1, BLOCK_SIZE] }> = c1.broadcast(tile_shape);
        let c2_v: Tile<f32, { [1, BLOCK_SIZE] }> = c2.broadcast(tile_shape);
        let mut dx_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> = dx.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let x_hat: Tile<f32, { [1, BLOCK_SIZE] }> = (tx - mean_v) * inv_std_v;
            let tdx: Tile<f32, { [1, BLOCK_SIZE] }> = inv_std_v * (tdy * tw - c1_v - x_hat * c2_v);
            dx_part.store(convert_tile(tdx), [0i32, j]);
        }
    }
}

pub use layer_norm_bwd_dx_f16_module::layer_norm_bwd_dx_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod layer_norm_bwd_dx_bf16_module {
    use cutile::core::*;

    /// Computes LayerNorm input gradient `dx` (bf16) over the last dimension [tokens, N].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn layer_norm_bwd_dx_bf16<const N: i32, const BLOCK_SIZE: i32>(
        dy: &Tensor<bf16, { [-1, N] }>,
        x: &Tensor<bf16, { [-1, N] }>,
        w: &Tensor<bf16, { [N] }>,
        dx: &mut Tensor<bf16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let dy_part: Partition<bf16, { [1, BLOCK_SIZE] }> = dy.partition(tile_shape);
        let x_part: Partition<bf16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let w_part: Partition<bf16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);

        let mut sum_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_x2: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        let mut sum_dy_w_x: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let dy_w: Tile<f32, { [1, BLOCK_SIZE] }> = tdy * tw;
            sum_x = sum_x + tx;
            sum_x2 = sum_x2 + tx * tx;
            sum_dy_w = sum_dy_w + dy_w;
            sum_dy_w_x = sum_dy_w_x + dy_w * tx;
        }
        let n: f32 = convert_scalar(N);
        let sum_x_1d: Tile<f32, { [1] }> = reduce_sum(sum_x, 1i32);
        let sum_x2_1d: Tile<f32, { [1] }> = reduce_sum(sum_x2, 1i32);
        let sum_dy_w_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w, 1i32);
        let sum_dy_w_x_1d: Tile<f32, { [1] }> = reduce_sum(sum_dy_w_x, 1i32);

        let sum_x_s: f32 = tile_to_scalar(sum_x_1d.reshape(const_shape![]));
        let sum_x2_s: f32 = tile_to_scalar(sum_x2_1d.reshape(const_shape![]));
        let sum_dy_w_s: f32 = tile_to_scalar(sum_dy_w_1d.reshape(const_shape![]));
        let sum_dy_w_x_s: f32 = tile_to_scalar(sum_dy_w_x_1d.reshape(const_shape![]));
        let mean: f32 = sum_x_s / n;
        let ex2: f32 = sum_x2_s / n;
        let var: f32 = ex2 - mean * mean;
        let inv_std_t: Tile<f32, { [] }> = rsqrt(scalar_to_tile(var + eps), ftz::Disabled);
        let inv_std: f32 = tile_to_scalar(inv_std_t);

        let c1: f32 = sum_dy_w_s / n;
        let dot_x: f32 = sum_dy_w_x_s / n;
        let c2: f32 = (dot_x - mean * c1) * inv_std;

        let mean_v: Tile<f32, { [1, BLOCK_SIZE] }> = mean.broadcast(tile_shape);
        let inv_std_v: Tile<f32, { [1, BLOCK_SIZE] }> = inv_std.broadcast(tile_shape);
        let c1_v: Tile<f32, { [1, BLOCK_SIZE] }> = c1.broadcast(tile_shape);
        let c2_v: Tile<f32, { [1, BLOCK_SIZE] }> = c2.broadcast(tile_shape);
        let mut dx_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> = dx.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tdy: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(dy_part.load([row, j]));
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(x_part.load([row, j]));
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> =
                convert_tile(w_part.load([j]).reshape(tile_shape));
            let x_hat: Tile<f32, { [1, BLOCK_SIZE] }> = (tx - mean_v) * inv_std_v;
            let tdx: Tile<f32, { [1, BLOCK_SIZE] }> = inv_std_v * (tdy * tw - c1_v - x_hat * c2_v);
            dx_part.store(convert_tile(tdx), [0i32, j]);
        }
    }
}

pub use layer_norm_bwd_dx_bf16_module::layer_norm_bwd_dx_bf16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rope_seq_bwd_f16_module {
    use cutile::core::*;

    /// Applies inverse rotary position embedding (backward pass) for f16 gradients shaped [seq_len, heads, D].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_bwd_f16<const D: i32, const HALF_D: i32>(
        dy: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        dx: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        let dy_part: Partition<f16, { [1, 1, HALF_D] }> = dy.partition(const_shape![1, 1, HALF_D]);
        let dy_lo: Tile<f32, { [1, 1, HALF_D] }> =
            convert_tile(dy_part.load([seq_idx, head_idx, 0i32]));
        let dy_hi: Tile<f32, { [1, 1, HALF_D] }> =
            convert_tile(dy_part.load([seq_idx, head_idx, 1i32]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos: f32 = convert_scalar(position_start + seq_idx);
        let theta: Tile<f32, { [1, 1, HALF_D] }> =
            (pos.broadcast(const_shape![HALF_D]) * freq).reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let dx_lo: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(dy_lo * cos_t + dy_hi * sin_t);
        let dx_hi: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(dy_hi * cos_t - dy_lo * sin_t);

        if half_idx == 0i32 {
            dx.store(dx_lo);
        } else {
            dx.store(dx_hi);
        }
    }
}

pub use rope_seq_bwd_f16_module::rope_seq_bwd_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rope_seq_bwd_bf16_module {
    use cutile::core::*;

    /// Applies inverse rotary position embedding (backward pass) for bf16 gradients shaped [seq_len, heads, D].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_bwd_bf16<const D: i32, const HALF_D: i32>(
        dy: &Tensor<bf16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        dx: &mut Tensor<bf16, { [1, 1, HALF_D] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        let dy_part: Partition<bf16, { [1, 1, HALF_D] }> = dy.partition(const_shape![1, 1, HALF_D]);
        let dy_lo: Tile<f32, { [1, 1, HALF_D] }> =
            convert_tile(dy_part.load([seq_idx, head_idx, 0i32]));
        let dy_hi: Tile<f32, { [1, 1, HALF_D] }> =
            convert_tile(dy_part.load([seq_idx, head_idx, 1i32]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos: f32 = convert_scalar(position_start + seq_idx);
        let theta: Tile<f32, { [1, 1, HALF_D] }> =
            (pos.broadcast(const_shape![HALF_D]) * freq).reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let dx_lo: Tile<bf16, { [1, 1, HALF_D] }> = convert_tile(dy_lo * cos_t + dy_hi * sin_t);
        let dx_hi: Tile<bf16, { [1, 1, HALF_D] }> = convert_tile(dy_hi * cos_t - dy_lo * sin_t);

        if half_idx == 0i32 {
            dx.store(dx_lo);
        } else {
            dx.store(dx_hi);
        }
    }
}

pub use rope_seq_bwd_bf16_module::rope_seq_bwd_bf16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod silu_mul_bwd_2d_f16_module {
    use cutile::core::*;

    /// Backward pass for fused SwiGLU `y = SiLU(gate) * up` (f16), writing `d_gate` and `d_up`.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn silu_mul_bwd_2d_f16<const BLOCK_SIZE: i32>(
        d_gate: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        d_up: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        dy: &Tensor<f16, { [-1, -1] }>,
        gate: &Tensor<f16, { [-1, -1] }>,
        up: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let shape = const_shape![1, BLOCK_SIZE];
        let dy_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(dy.partition(shape).load([row, col]));
        let gate_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(gate.partition(shape).load([row, col]));
        let up_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(up.partition(shape).load([row, col]));
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, shape);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, shape);
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, one + exp(zero - gate_f32));
        let silu: Tile<f32, { [1, BLOCK_SIZE] }> = gate_f32 * sigmoid;
        let dsilu: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * (one + gate_f32 * (one - sigmoid));
        d_up.store(convert_tile(dy_f32 * silu));
        d_gate.store(convert_tile(dy_f32 * up_f32 * dsilu));
    }
}

pub use silu_mul_bwd_2d_f16_module::silu_mul_bwd_2d_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod silu_mul_bwd_2d_bf16_module {
    use cutile::core::*;

    /// Backward pass for fused SwiGLU `y = SiLU(gate) * up` (bf16), writing `d_gate` and `d_up`.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn silu_mul_bwd_2d_bf16<const BLOCK_SIZE: i32>(
        d_gate: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        d_up: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        dy: &Tensor<bf16, { [-1, -1] }>,
        gate: &Tensor<bf16, { [-1, -1] }>,
        up: &Tensor<bf16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let shape = const_shape![1, BLOCK_SIZE];
        let dy_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(dy.partition(shape).load([row, col]));
        let gate_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(gate.partition(shape).load([row, col]));
        let up_f32: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(up.partition(shape).load([row, col]));
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, shape);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, shape);
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, one + exp(zero - gate_f32));
        let silu: Tile<f32, { [1, BLOCK_SIZE] }> = gate_f32 * sigmoid;
        let dsilu: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * (one + gate_f32 * (one - sigmoid));
        d_up.store(convert_tile(dy_f32 * silu));
        d_gate.store(convert_tile(dy_f32 * up_f32 * dsilu));
    }
}

pub use silu_mul_bwd_2d_bf16_module::silu_mul_bwd_2d_bf16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod cross_entropy_fwd_bwd_bf16_module {
    use cutile::core::*;

    /// Fused online-softmax cross-entropy forward + backward over [tokens, V] for bf16 logits.
    /// Each CTA handles one token row, writing `loss[row]` (f32) and `d_logits[row, :]` (bf16).
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn cross_entropy_fwd_bwd_bf16<const V: i32, const BLOCK_SIZE: i32>(
        logits: &Tensor<bf16, { [-1, V] }>,
        targets: &Tensor<i32, { [-1] }>,
        d_logits: &mut Tensor<bf16, { [1, V] }>,
        losses: &mut Tensor<f32, { [1] }>,
        grad_scale: f32,
        ignore_index: i32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = V / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let target_t: Tile<i32, { [1] }> = targets.partition(const_shape![1]).load([row]);
        let target: i32 = tile_to_scalar(target_t.reshape(const_shape![]));

        let logits_part: Partition<bf16, { [1, BLOCK_SIZE] }> = logits.partition(tile_shape);
        let mut d_logits_part: PartitionMut<bf16, { [1, BLOCK_SIZE] }> =
            d_logits.partition_mut(tile_shape);

        if target == ignore_index {
            let zero_bf16: Tile<bf16, { [1, BLOCK_SIZE] }> =
                convert_tile(constant(0.0f32, tile_shape));
            for j in 0i32..num_tiles {
                d_logits_part.store(zero_bf16, [0i32, j]);
            }
            losses.store(constant(0.0f32, const_shape![1]));
        } else {
            // Pass 1: online softmax (m, l) across vocabulary tiles
            let mut m: f32 = 0.0f32 - 1.0e30f32;
            let mut l: f32 = 0.0f32;
            let mut target_logit: f32 = 0.0f32;
            let arange: Tile<i32, { [1, BLOCK_SIZE] }> =
                iota(const_shape![BLOCK_SIZE]).reshape(tile_shape);
            let target_v: Tile<i32, { [1, BLOCK_SIZE] }> = target.broadcast(tile_shape);
            let zero_v: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, tile_shape);

            for j in 0i32..num_tiles {
                let z: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(logits_part.load([row, j]));
                let tile_max_1d: Tile<f32, { [1] }> = reduce_max(z, 1i32);
                let tile_max: f32 = tile_to_scalar(tile_max_1d.reshape(const_shape![]));
                let m_new: f32 = max(m, tile_max);
                let exp_diff_t: Tile<f32, { [] }> = exp(scalar_to_tile(m - m_new));
                let alpha: f32 = tile_to_scalar(exp_diff_t);
                let p: Tile<f32, { [1, BLOCK_SIZE] }> = exp(z - m_new.broadcast(tile_shape));
                let tile_sum_1d: Tile<f32, { [1] }> = reduce_sum(p, 1i32);
                let tile_sum: f32 = tile_to_scalar(tile_sum_1d.reshape(const_shape![]));
                l = l * alpha + tile_sum;
                m = m_new;

                let base: Tile<i32, { [1, BLOCK_SIZE] }> = (j * BLOCK_SIZE).broadcast(tile_shape);
                let idx: Tile<i32, { [1, BLOCK_SIZE] }> = base + arange;
                let is_target: Tile<bool, { [1, BLOCK_SIZE] }> = eq_tile(idx, target_v);
                let picked: Tile<f32, { [1, BLOCK_SIZE] }> = select(is_target, z, zero_v);
                let picked_1d: Tile<f32, { [1] }> = reduce_sum(picked, 1i32);
                let picked_s: f32 = tile_to_scalar(picked_1d.reshape(const_shape![]));
                target_logit = target_logit + picked_s;
            }

            let log_l_t: Tile<f32, { [] }> = log(scalar_to_tile(l));
            let log_l: f32 = tile_to_scalar(log_l_t);
            let loss_val: f32 = log_l + m - target_logit;
            losses.store(loss_val.broadcast(const_shape![1]));

            // Pass 2: write d_logits = grad_scale * (softmax(z) - 1[idx == target])
            let m_v: Tile<f32, { [1, BLOCK_SIZE] }> = m.broadcast(tile_shape);
            let inv_l_v: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 / l).broadcast(tile_shape);
            let scale_v: Tile<f32, { [1, BLOCK_SIZE] }> = grad_scale.broadcast(tile_shape);
            let one_v: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, tile_shape);

            for j in 0i32..num_tiles {
                let z: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(logits_part.load([row, j]));
                let prob: Tile<f32, { [1, BLOCK_SIZE] }> = exp(z - m_v) * inv_l_v;
                let base: Tile<i32, { [1, BLOCK_SIZE] }> = (j * BLOCK_SIZE).broadcast(tile_shape);
                let idx: Tile<i32, { [1, BLOCK_SIZE] }> = base + arange;
                let is_target: Tile<bool, { [1, BLOCK_SIZE] }> = eq_tile(idx, target_v);
                let indicator: Tile<f32, { [1, BLOCK_SIZE] }> = select(is_target, one_v, zero_v);
                let dz: Tile<f32, { [1, BLOCK_SIZE] }> = (prob - indicator) * scale_v;
                d_logits_part.store(convert_tile(dz), [0i32, j]);
            }
        }
    }
}

pub use cross_entropy_fwd_bwd_bf16_module::cross_entropy_fwd_bwd_bf16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod fused_adamw_bf16_module {
    use cutile::core::*;

    /// Fused decoupled weight-decay AdamW update for bf16 parameters and moments.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn fused_adamw_bf16<const BLOCK_SIZE: i32>(
        param_out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        exp_avg_out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        exp_avg_sq_out: &mut Tensor<bf16, { [1, BLOCK_SIZE] }>,
        param: &Tensor<bf16, { [-1, -1] }>,
        grad: &Tensor<bf16, { [-1, -1] }>,
        exp_avg: &Tensor<bf16, { [-1, -1] }>,
        exp_avg_sq: &Tensor<bf16, { [-1, -1] }>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let shape = const_shape![1, BLOCK_SIZE];

        let p: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(param.partition(shape).load([row, col]));
        let g: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(grad.partition(shape).load([row, col]));
        let m: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(exp_avg.partition(shape).load([row, col]));
        let v: Tile<f32, { [1, BLOCK_SIZE] }> =
            convert_tile(exp_avg_sq.partition(shape).load([row, col]));

        let b1: Tile<f32, { [1, BLOCK_SIZE] }> = beta1.broadcast(shape);
        let one_minus_b1: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 - beta1).broadcast(shape);
        let b2: Tile<f32, { [1, BLOCK_SIZE] }> = beta2.broadcast(shape);
        let one_minus_b2: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 - beta2).broadcast(shape);

        let m_new: Tile<f32, { [1, BLOCK_SIZE] }> = b1 * m + one_minus_b1 * g;
        let v_new: Tile<f32, { [1, BLOCK_SIZE] }> = b2 * v + one_minus_b2 * (g * g);

        let inv_bc1: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 / bias_correction1).broadcast(shape);
        let inv_bc2: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 / bias_correction2).broadcast(shape);
        let m_hat: Tile<f32, { [1, BLOCK_SIZE] }> = m_new * inv_bc1;
        let v_hat: Tile<f32, { [1, BLOCK_SIZE] }> = v_new * inv_bc2;

        let denom: Tile<f32, { [1, BLOCK_SIZE] }> =
            sqrt(v_hat, rounding::NearestEven, ftz::Disabled) + eps.broadcast(shape);
        let step_dir: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(m_hat, denom);

        let decay: Tile<f32, { [1, BLOCK_SIZE] }> = (1.0f32 - lr * weight_decay).broadcast(shape);
        let lr_v: Tile<f32, { [1, BLOCK_SIZE] }> = lr.broadcast(shape);
        let p_new: Tile<f32, { [1, BLOCK_SIZE] }> = p * decay - lr_v * step_dir;

        param_out.store(convert_tile(p_new));
        exp_avg_out.store(convert_tile(m_new));
        exp_avg_sq_out.store(convert_tile(v_new));
    }
}

pub use fused_adamw_bf16_module::fused_adamw_bf16;
