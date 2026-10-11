/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stack slots for values the lowering must route through memory.
//!
//! Some MIR values need an address while being lowered: a borrowed SSA value
//! (`mir.ref`), an enum whose payload field has no struct slot of its own, an
//! array indexed at run time, a transmute between unrelated layouts. Each gets
//! a one-element `alloca`, a store at the use site, and loads through it.
//!
//! The `alloca` itself always goes to the function's entry block. LLVM's SROA
//! and mem2reg only promote static allocas, which must sit there; a slot
//! created where the value is used (an enum match inside a loop) survives
//! `opt`, NVPTX hoists it into the local-memory frame, and every access becomes
//! a local load or store. In the entry block the same slot is promoted back to
//! registers whenever its accesses allow it.

use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::{Inserter, OpInsertionPoint};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Allocate one `ty` at the start of the entry block of the function being
/// rewritten and return its pointer. The rewriter's insertion point is left
/// where it was, so the caller emits its store and loads at the use site.
///
/// `align` is the ABI alignment to record on the slot; `None` keeps LLVM's
/// default for `ty`.
pub(crate) fn entry_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: TypeHandle,
    align: Option<u64>,
) -> Value {
    let entry = entry_block(ctx, rewriter);
    let saved = rewriter.get_insertion_point();
    // Both ops go in front of the block's current first op, in this order, so
    // the count constant precedes the alloca that uses it.
    rewriter.set_insertion_point(match entry.deref(ctx).iter(ctx).next() {
        Some(first) => OpInsertionPoint::BeforeOperation(first),
        None => OpInsertionPoint::AtBlockEnd(entry),
    });

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one = IntegerAttr::new(i32_ty, APInt::from_i64(1, NonZeroUsize::new(32).unwrap()));
    let count = llvm::ConstantOp::new(ctx, Box::new(one));
    rewriter.insert_operation(ctx, count.get_operation());
    let count = count.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, ty, count, 0);
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, alloca.get_operation());
    rewriter.set_insertion_point(saved);
    alloca.get_operation().deref(ctx).get_result(0)
}

/// Entry block of the function region the rewriter is currently inserting into.
fn entry_block(ctx: &Context, rewriter: &DialectConversionRewriter) -> Ptr<BasicBlock> {
    let block = rewriter
        .get_insertion_point()
        .get_insertion_block(ctx)
        .expect("lowering inserts into a function body block");
    let region = block
        .deref(ctx)
        .get_parent_region()
        .expect("a function body block belongs to a region");
    region
        .deref(ctx)
        .get_head()
        .expect("a region holding the insertion block has an entry block")
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use crate::convert::ops::test_util::*;
    use dialect_mir::ops as mir;
    use dialect_mir::types::MirArrayType;
    use llvm_export::ops as llvm;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::op::Op;
    use pliron::operation::Operation;
    use pliron::r#type::TypeHandle;

    /// A slot needed in a later block (here: a runtime-indexed array read after a
    /// branch) is still allocated in the function's entry block, where SROA can
    /// promote it; only its store and loads stay in the later block.
    #[test]
    fn slot_of_a_later_block_is_allocated_in_the_entry_block() {
        let mut ctx = make_ctx();
        let element: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let index_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let array_ty: TypeHandle = MirArrayType::get(&mut ctx, element, 3).into();
        let (module, entry) = build_kernel(&mut ctx, vec![index_ty], vec![element]);
        let index = entry.deref(&ctx).get_argument(0);

        let later = append_block(&mut ctx, entry, vec![index_ty]);
        let later_index = later.deref(&ctx).get_argument(0);
        let undef = mir::MirUndefOp::new(&mut ctx, array_ty);
        undef.get_operation().insert_at_back(later, &ctx);
        let array = undef.get_operation().deref(&ctx).get_result(0);
        let extract = Operation::new(
            &mut ctx,
            mir::MirExtractArrayElementOp::get_concrete_op_info(),
            vec![element],
            vec![array, later_index],
            vec![],
            0,
        );
        extract.insert_at_back(later, &ctx);
        let value = extract.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, later, vec![value]);

        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![index],
            vec![later],
            0,
        );
        goto.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");

        let blocks = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &blocks), 1);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &blocks[..1]),
            1,
            "the slot must be allocated in the entry block"
        );
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &[later]), 1);
    }
}
