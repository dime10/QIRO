//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file was modified for the QMLIR project.
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_PASSDETAIL_H_
#define TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace linalg {
class LinalgDialect;
} // end namespace linalg

template <typename DerivedT>
class InlinerBase : public ::mlir::OperationPass<> {
public:
  InlinerBase() : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}
  InlinerBase(const InlinerBase &) : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}

  /// Returns the command-line argument attached to this pass.
  ::llvm::StringRef getArgument() const override { return "inline"; }

  /// Returns the derived pass name.
  ::llvm::StringRef getName() const override { return "Inliner"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {

  }

protected:
  ::mlir::Pass::Option<bool> disableCanonicalization{*this, "disable-simplify", ::llvm::cl::desc("Disable running simplifications during inlining"), ::llvm::cl::init(false)};
  ::mlir::Pass::Option<unsigned> maxInliningIterations{*this, "max-iterations", ::llvm::cl::desc("Maximum number of iterations when inlining within an SCC"), ::llvm::cl::init(4)};
};

} // end namespace mlir

#endif // TRANSFORMS_PASSDETAIL_H_
