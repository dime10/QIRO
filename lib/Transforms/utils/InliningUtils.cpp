//===- InliningUtils.cpp ---- Misc utilities for inlining -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous inlining utilities.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file was modified for the QMLIR project.
//===----------------------------------------------------------------------===//

#include "InliningUtils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "QuantumDialect.h"
#include "QuantumSSADialect.h"

#define DEBUG_TYPE "inlining"

using namespace mlir;

/// Remap locations from the inlined blocks with CallSiteLoc locations with the
/// provided caller location.
static void
remapInlinedLocations(iterator_range<Region::iterator> inlinedBlocks,
                      Location callerLoc) {
  DenseMap<Location, Location> mappedLocations;
  auto remapOpLoc = [&](Operation *op) {
    auto it = mappedLocations.find(op->getLoc());
    if (it == mappedLocations.end()) {
      auto newLoc = CallSiteLoc::get(op->getLoc(), callerLoc);
      it = mappedLocations.try_emplace(op->getLoc(), newLoc).first;
    }
    op->setLoc(it->second);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOpLoc);
}

static void remapInlinedOperands(iterator_range<Region::iterator> inlinedBlocks,
                                 BlockAndValueMapping &mapper) {
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOperands);
}

//===----------------------------------------------------------------------===//
// Intercept calls to InlinerInterface
//===----------------------------------------------------------------------===//

static bool isOpLegalToInline(InlinerInterface &interface, Operation *op, Region *insertRegion,
                              BlockAndValueMapping &valueMapping) {
  // bypass affine ops only inlining into function ops
  if (isa<AffineForOp>(op) || isa<AffineYieldOp>(op))
    return true;
  return interface.isLegalToInline(op, insertRegion, valueMapping);
}

static bool isRegionLegalToInline(InlinerInterface &interface, Region *insertRegion, Region *src,
                                  BlockAndValueMapping &valueMapping) {
  // only inline circuits
  if (!isa<quantum::CircuitOp>(src->getParentOp()) && !isa<quantumssa::CircuitOp>(src->getParentOp()))
    return false;
  // bypass affine dialect not allowing inlining into its ops
  else if (isa<AffineForOp>(insertRegion->getParentOp()))
    return true;
  // SCF have no inlining restrictions

  // also check parent circuit region if any
  if (auto circ = insertRegion->getParentOfType<quantum::CircuitOp>()) {
    if (!interface.isLegalToInline(&circ.getRegion(), src, valueMapping))
      return false;
  } else if (auto circ = insertRegion->getParentOfType<quantumssa::CircuitOp>()) {
    if (!interface.isLegalToInline(&circ.getRegion(), src, valueMapping))
      return false;
  }

  // refer to interface for all other cases
  return interface.isLegalToInline(insertRegion, src, valueMapping);
}

/// Utility to check that all of the operations within 'src' can be inlined.
static bool isLegalToInline(InlinerInterface &interface, Region *src,
                            Region *insertRegion,
                            BlockAndValueMapping &valueMapping) {
  for (auto &block : *src) {
    for (auto &op : block) {
      // Check this operation.
      if (!isOpLegalToInline(interface, &op, insertRegion, valueMapping)) {
        LLVM_DEBUG({
          llvm::dbgs() << "* Illegal to inline because of op: ";
          op.dump();
        });
        return false;
      }
      // Check any nested regions.
      if (interface.shouldAnalyzeRecursively(&op) &&
          llvm::any_of(op.getRegions(), [&](Region &region) {
            return !isLegalToInline(interface, &region, insertRegion,
                                    valueMapping);
          }))
        return false;
    }
  }
  return true;
}

static void propComputeTags(Builder &b, const llvm::iterator_range<mlir::Region::iterator> &blocks,
                            bool comp, bool uncomp) {
  for (auto &block : blocks) {
    for (auto &op : block) {
      if (isa<mlir::scf::ForOp>(op) || isa<mlir::scf::IfOp>(op))
        for (auto &region : op.getRegions())
          propComputeTags(b, region.getBlocks(), comp, uncomp);
      if (comp)
        op.setAttr("compute", b.getUnitAttr());
      if (uncomp)
        op.setAttr("uncompute", b.getUnitAttr());
    }
  }
}

//===----------------------------------------------------------------------===//
// Inline Methods
//===----------------------------------------------------------------------===//

LogicalResult mlir::quantum::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 BlockAndValueMapping &mapper,
                                 ValueRange resultsToReplace,
                                 TypeRange regionResultTypes,
                                 Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  assert(resultsToReplace.size() == regionResultTypes.size());
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  // Check that all of the region arguments have been mapped.
  auto *srcEntryBlock = &src->front();
  if (llvm::any_of(srcEntryBlock->getArguments(),
                   [&](BlockArgument arg) { return !mapper.contains(arg); }))
    return failure();

  // The insertion point must be within a block.
  Block *insertBlock = inlinePoint->getBlock();
  if (!insertBlock)
    return failure();
  Region *insertRegion = insertBlock->getParent();

  // Check that the operations within the source region are valid to inline.
  if (!isRegionLegalToInline(interface, insertRegion, src, mapper) ||
      !isLegalToInline(interface, src, insertRegion, mapper))
    return failure();

  // Split the insertion block.
  Block *postInsertBlock =
      insertBlock->splitBlock(++inlinePoint->getIterator());

  // Check to see if the region is being cloned, or moved inline. In either
  // case, move the new blocks after the 'insertBlock' to improve IR
  // readability.
  if (shouldCloneInlinedRegion)
    src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
  else
    insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                     src->getBlocks(), src->begin(),
                                     src->end());

  // Get the range of newly inserted blocks.
  auto newBlocks = llvm::make_range(std::next(insertBlock->getIterator()),
                                    postInsertBlock->getIterator());
  Block *firstNewBlock = &*newBlocks.begin();

  // propagate compute uncompute tags
  Builder b(inlinePoint->getContext());
  propComputeTags(b, newBlocks, !!inlinePoint->getAttr("compute"), !!inlinePoint->getAttr("uncompute"));

  // Remap the locations of the inlined operations if a valid source location
  // was provided.
  if (inlineLoc && !inlineLoc->isa<UnknownLoc>())
    remapInlinedLocations(newBlocks, *inlineLoc);

  // If the blocks were moved in-place, make sure to remap any necessary
  // operands.
  if (!shouldCloneInlinedRegion)
    remapInlinedOperands(newBlocks, mapper);

  // Process the newly inlined blocks.
  interface.processInlinedBlocks(newBlocks);

  // Handle the case where only a single block was inlined.
  if (std::next(newBlocks.begin()) == newBlocks.end()) {
    // Have the interface handle the terminator of this block.
    auto *firstBlockTerminator = firstNewBlock->getTerminator();
    interface.handleTerminator(firstBlockTerminator,
                               llvm::to_vector<6>(resultsToReplace));
    firstBlockTerminator->erase();

    // Merge the post insert block into the cloned entry block.
    firstNewBlock->getOperations().splice(firstNewBlock->end(),
                                          postInsertBlock->getOperations());
    postInsertBlock->erase();
  } else {
    // Otherwise, there were multiple blocks inlined. Add arguments to the post
    // insertion block to represent the results to replace.
    for (auto resultToRepl : llvm::enumerate(resultsToReplace)) {
      resultToRepl.value().replaceAllUsesWith(postInsertBlock->addArgument(
          regionResultTypes[resultToRepl.index()]));
    }

    /// Handle the terminators for each of the new blocks.
    for (auto &newBlock : newBlocks)
      interface.handleTerminator(newBlock.getTerminator(), postInsertBlock);
  }

  // Splice the instructions of the inlined entry block into the insert block.
  insertBlock->getOperations().splice(insertBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
  return success();
}

/// This function is an overload of the above 'inlineRegion' that allows for
/// providing the set of operands ('inlinedOperands') that should be used
/// in-favor of the region arguments when inlining.
LogicalResult mlir::quantum::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 ValueRange inlinedOperands,
                                 ValueRange resultsToReplace,
                                 Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  auto *entryBlock = &src->front();
  if (inlinedOperands.size() != entryBlock->getNumArguments())
    return failure();

  // Map the provided call operands to the arguments of the region.
  BlockAndValueMapping mapper;
  for (unsigned i = 0, e = inlinedOperands.size(); i != e; ++i) {
    // Verify that the types of the provided values match the function argument
    // types.
    BlockArgument regionArg = entryBlock->getArgument(i);
    if (inlinedOperands[i].getType() != regionArg.getType())
      return failure();
    mapper.map(regionArg, inlinedOperands[i]);
  }

  // Call into the main region inliner function.
  return inlineRegion(interface, src, inlinePoint, mapper, resultsToReplace,
                      resultsToReplace.getTypes(), inlineLoc,
                      shouldCloneInlinedRegion);
}

/// Utility function used to generate a cast operation from the given interface,
/// or return nullptr if a cast could not be generated.
static Value materializeConversion(const DialectInlinerInterface *interface,
                                   SmallVectorImpl<Operation *> &castOps,
                                   OpBuilder &castBuilder, Value arg, Type type,
                                   Location conversionLoc) {
  if (!interface)
    return nullptr;

  // Check to see if the interface for the call can materialize a conversion.
  Operation *castOp = interface->materializeCallConversion(castBuilder, arg,
                                                           type, conversionLoc);
  if (!castOp)
    return nullptr;
  castOps.push_back(castOp);

  // Ensure that the generated cast is correct.
  assert(castOp->getNumOperands() == 1 && castOp->getOperand(0) == arg &&
         castOp->getNumResults() == 1 && *castOp->result_type_begin() == type);
  return castOp->getResult(0);
}

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult mlir::quantum::inlineCall(InlinerInterface &interface,
                               CallOpInterface call,
                               CallableOpInterface callable, Region *src,
                               bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();
  auto *entryBlock = &src->front();
  ArrayRef<Type> callableResultTypes = callable.getCallableResults();

  // Make sure that the number of arguments and results matchup between the call
  // and the region.
  SmallVector<Value, 8> callOperands(call.getArgOperands());
  SmallVector<Value, 8> callResults(call.getOperation()->getResults());
  if (callOperands.size() != entryBlock->getNumArguments() ||
      callResults.size() != callableResultTypes.size())
    return failure();

  // A set of cast operations generated to matchup the signature of the region
  // with the signature of the call.
  SmallVector<Operation *, 4> castOps;
  castOps.reserve(callOperands.size() + callResults.size());

  // Functor used to cleanup generated state on failure.
  auto cleanupState = [&] {
    for (auto *op : castOps) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
    return failure();
  };

  // Builder used for any conversion operations that need to be materialized.
  OpBuilder castBuilder(call);
  Location castLoc = call.getLoc();
  auto *callInterface = interface.getInterfaceFor(call.getDialect());

  // Map the provided call operands to the arguments of the region.
  BlockAndValueMapping mapper;
  for (unsigned i = 0, e = callOperands.size(); i != e; ++i) {
    BlockArgument regionArg = entryBlock->getArgument(i);
    Value operand = callOperands[i];

    // If the call operand doesn't match the expected region argument, try to
    // generate a cast.
    Type regionArgType = regionArg.getType();
    if (operand.getType() != regionArgType) {
      if (!(operand = materializeConversion(callInterface, castOps, castBuilder,
                                            operand, regionArgType, castLoc)))
        return cleanupState();
    }
    mapper.map(regionArg, operand);
  }

  // Ensure that the resultant values of the call match the callable.
  castBuilder.setInsertionPointAfter(call);
  for (unsigned i = 0, e = callResults.size(); i != e; ++i) {
    Value callResult = callResults[i];
    if (callResult.getType() == callableResultTypes[i])
      continue;

    // Generate a conversion that will produce the original type, so that the IR
    // is still valid after the original call gets replaced.
    Value castResult =
        materializeConversion(callInterface, castOps, castBuilder, callResult,
                              callResult.getType(), castLoc);
    if (!castResult)
      return cleanupState();
    callResult.replaceAllUsesWith(castResult);
    castResult.getDefiningOp()->replaceUsesOfWith(castResult, callResult);
  }

  // Attempt to inline the call.
  if (failed(inlineRegion(interface, src, call, mapper, callResults,
                          callableResultTypes, call.getLoc(),
                          shouldCloneInlinedRegion)))
    return cleanupState();
  return success();
}
