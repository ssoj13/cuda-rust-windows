/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device math that avoids local memory.
//!
//! `f32::sin` / `f32::cos` lower to libdevice `__nv_sinf` / `__nv_cosf`. Their
//! argument reduction has two paths: a three-constant Cody-Waite reduction for
//! `|x| < 105615`, and a Payne-Hanek reduction for larger arguments that keeps
//! a seven-word table in local memory. Every inlined call carries both, so a
//! kernel that only ever sees small angles still gets a local-memory frame,
//! local loads and stores, and several hundred instructions per call site.
//!
//! [`sin_cos`] is the first path alone: the same constants, the same
//! correctly-rounded operations in the same order, hence the same bits as
//! libdevice for every argument it accepts. One reduction serves both results.

use crate::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};

/// `2/pi`, rounded (libdevice `0x3F22F983`).
const TWO_OVER_PI: f32 = f32::from_bits(0x3F22_F983);
/// `pi/2` split in three parts for the Cody-Waite reduction (negated, libdevice order).
const NEG_PI_2_HI: f32 = f32::from_bits(0xBFC9_0FDA);
const NEG_PI_2_MID: f32 = f32::from_bits(0xB3A2_2168);
const NEG_PI_2_LO: f32 = f32::from_bits(0xA7C2_34C5);
/// `1.5 * 2^23`: adding it rounds an `f32` below `2^22` in magnitude to the nearest
/// integer (ties to even) and leaves that integer in the low mantissa bits.
const ROUNDING_SHIFTER: f32 = f32::from_bits(0x4B40_0000);
/// Largest argument magnitude the Cody-Waite path covers (libdevice switches at it).
pub const SIN_COS_MAX_ARG: f32 = 105_615.0;

/// `(sin x, cos x)`, bit-identical to libdevice `__nv_sinf` / `__nv_cosf` (non-FTZ)
/// for `|x| < SIN_COS_MAX_ARG`, without their local-memory slow path.
///
/// The caller guarantees the bound; every angle derived from a direction, a few
/// turns of rotation or a sampled `2 * pi * u` satisfies it. Beyond it the
/// reduction is not exact and accuracy degrades with `|x|` instead of switching
/// to the Payne-Hanek path. NaN maps to NaN.
#[inline(always)]
pub fn sin_cos(x: f32) -> (f32, f32) {
    // q = round(x * 2/pi): libdevice uses cvt.rni; the shifter gives the same
    // integer for |x * 2/pi| < 2^22, and its low bits are the quadrant.
    let shifted = add_rn_f32(mul_rn_f32(x, TWO_OVER_PI), ROUNDING_SHIFTER);
    let q = add_rn_f32(shifted, -ROUNDING_SHIFTER);
    let quadrant = shifted.to_bits();
    let t = fma_rn_f32(q, NEG_PI_2_HI, x);
    let t = fma_rn_f32(q, NEG_PI_2_MID, t);
    let t = fma_rn_f32(q, NEG_PI_2_LO, t);
    (
        sin_reduced(t, quadrant),
        sin_reduced(t, quadrant.wrapping_add(1)),
    )
}

/// libdevice `__internal_accurate_sinf` after the reduction: `sin(t + quadrant * pi/2)`
/// for `|t| <= pi/4`, an odd (sine) or even (cosine) minimax polynomial by the
/// quadrant's low bit, negated when its second bit is set.
#[inline(always)]
fn sin_reduced(t: f32, quadrant: u32) -> f32 {
    let s = mul_rn_f32(t, t);
    let cosine = quadrant & 1 != 0;
    let lead = if cosine { 1.0 } else { t };
    let lead_s = fma_rn_f32(s, lead, 0.0);
    let p = if cosine {
        fma_rn_f32(f32::from_bits(0x37CB_AC00), s, f32::from_bits(0xBAB6_07ED))
    } else {
        f32::from_bits(0xB94D_4153)
    };
    let p = fma_rn_f32(
        p,
        s,
        f32::from_bits(if cosine { 0x3D2A_AABB } else { 0x3C08_85E4 }),
    );
    let p = fma_rn_f32(
        p,
        s,
        f32::from_bits(if cosine { 0xBEFF_FFFF } else { 0xBE2A_AAA8 }),
    );
    let r = fma_rn_f32(p, lead_s, lead);
    // libdevice negates with fma(r, -1, 0), which keeps +0 for r = +0.
    if quadrant & 2 != 0 {
        fma_rn_f32(r, -1.0, 0.0)
    } else {
        r
    }
}
