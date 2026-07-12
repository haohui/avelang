#!/usr/bin/env python3
import re
import os
import shutil
import subprocess
import unittest

import _avelang_bindings as _C
import avelang
import avelang.language as S
from avelang.language import invariant
from avelang.compiler.code_generator import (
    _build_import_module,
    _collect_jit_dependencies,
    _get_function_def,
)
from avelang.backends.amdgpu.compiler import AmdgpuCompiler
from avelang.backends.compiler import GPUTarget
from avelang.runtime.jit import compute_cache_key

Z3_BIN = shutil.which("z3")


def generate_mlir(jit_fn):
    jit_deps = _collect_jit_dependencies(jit_fn)
    import_module = _build_import_module([jit_fn, *jit_deps])

    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)

    for dep in jit_deps:
        dep_func = _get_function_def(dep.parse())
        generator.visit_function_def(dep_func, "[]", "jit")

    kernel_func = _get_function_def(jit_fn.parse())
    generator.visit_function_def(kernel_func, "[]", "kernel")
    return generator.get_mlir()


def _parse_string_array(text):
    return re.findall(r'"([^"]+)"', text)


def extract_tag_bindings(mlir):
    bindings = {}
    for match in re.finditer(r"ave\.tag\.bind\s+%\w+\s+\{([^}]*)\}", mlir):
        attrs = match.group(1)
        name_match = re.search(r'tag_name = "([^"]+)"', attrs)
        inputs_match = re.search(r"tag_inputs = \[([^\]]*)\]", attrs)
        exprs_match = re.search(r"tag_exprs = \[([^\]]*)\]", attrs)
        if not name_match or not inputs_match or not exprs_match:
            continue
        bindings[name_match.group(1)] = {
            "inputs": _parse_string_array(inputs_match.group(1)),
            "exprs": _parse_string_array(exprs_match.group(1)),
        }
    return bindings


def tag_expr_to_smt(expr):
    tokens = re.findall(r"\(|\)|[^\s()]+", expr)
    pos = 0

    def parse_expr():
        nonlocal pos
        token = tokens[pos]
        pos += 1
        if token != "(":
            return token

        op = tokens[pos]
        pos += 1
        args = []
        while tokens[pos] != ")":
            args.append(parse_expr())
        pos += 1

        if op == "add":
            return f"(+ {args[0]} {args[1]})"
        if op == "sub":
            return f"(- {args[0]} {args[1]})"
        if op == "mult":
            return f"(* {args[0]} {args[1]})"
        if op == "floordiv":
            return f"(div {args[0]} {args[1]})"
        if op == "mod":
            return f"(mod {args[0]} {args[1]})"
        if op == "-" and len(args) == 1:
            return f"(- {args[0]})"
        raise ValueError(f"unsupported tag op: {op}")

    result = parse_expr()
    if pos != len(tokens):
        raise ValueError(f"unexpected trailing tokens in tag expr: {expr}")
    return result


def run_z3(smtlib):
    if not Z3_BIN:
        raise unittest.SkipTest("z3 executable is not available")
    result = subprocess.run(
        [Z3_BIN, "-smt2", "-in"],
        input=smtlib,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[0]


@avelang.jit
def kernel_invariant_shared_copy(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((32,), S.i32),
):
    invariant.tag(input_data, lambda lane: (lane,), "LaneTag")

    shared_buf = S.make_shared((32,), S.i32)
    thread_id = S.thread_id(0)

    shared_buf[thread_id] = input_data[thread_id]
    S.syncthreads()

    value = shared_buf[thread_id]
    invariant.assert_tag_eq(value, input_data[thread_id])
    output_data[thread_id] = value


@avelang.jit
def kernel_invariant_reset_shared(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((32,), S.i32),
):
    invariant.tag(input_data, lambda lane: (lane,), "LaneTag")

    shared_buf = S.make_shared((32,), S.i32)
    thread_id = S.thread_id(0)

    shared_buf[thread_id] = input_data[thread_id]
    invariant.reset_tags(shared_buf)
    output_data[thread_id] = shared_buf[thread_id]


@avelang.jit
def kernel_invariant_loop_index(
    input_data: S.Tensor((4,), S.i32),
    output_data: S.Tensor((4,), S.i32),
):
    invariant.tag(input_data, lambda i: (i,), "IndexTag")

    for i in S.range(4):
        value = input_data[i]
        invariant.assert_tag_eq(value, input_data[i])
        output_data[i] = value


@avelang.jit
def kernel_invariant_z3_equivalent(
    lhs: S.Tensor((4,), S.i32),
    rhs: S.Tensor((4,), S.i32),
    output_data: S.Tensor((4,), S.i32),
):
    invariant.tag(lhs, lambda i: (i,), "BaseTag")
    invariant.tag(rhs, lambda j: (((j + 1) - 1),), "EquivalentTag")

    for i in S.range(4):
        left = lhs[i]
        right = rhs[i]
        invariant.assert_tag_eq(left, right)
        output_data[i] = left


@avelang.jit
def kernel_invariant_z3_mismatch(
    lhs: S.Tensor((4,), S.i32),
    rhs: S.Tensor((4,), S.i32),
    output_data: S.Tensor((4,), S.i32),
):
    invariant.tag(lhs, lambda i: (i,), "BaseTag")
    invariant.tag(rhs, lambda j: ((j + 1),), "ShiftedTag")

    for i in S.range(4):
        left = lhs[i]
        right = rhs[i]
        invariant.assert_tag_eq(left, right)
        output_data[i] = left


@avelang.jit
def kernel_invariant_raw_buffer_load(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((1,), S.i32),
):
    invariant.tag(input_data, lambda i: (i,), "RawBufferTag")
    rsrc = S.amdgpu.make_rsrc(input_data, S.convert(32, S.i32))
    value = S.amdgpu.raw_buffer_load_x1(
        rsrc, S.convert(0, S.i32), S.convert(4, S.i32), 0
    )
    invariant.assert_tag_eq(value, input_data[1])
    output_data[0] = value


@avelang.jit
def kernel_invariant_raw_buffer_load_lds(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((1,), S.i32),
):
    invariant.tag(input_data, lambda i: (i,), "RawBufferLdsTag")
    rsrc = S.amdgpu.make_rsrc(input_data, S.convert(32, S.i32))
    shared = S.make_shared((2,), S.i32)
    S.amdgpu.raw_buffer_load_x1_lds(
        rsrc, shared, 4, S.convert(0, S.i32), S.convert(4, S.i32),
        S.convert(4, S.i32), 0
    )
    value = shared[1]
    invariant.assert_tag_eq(value, input_data[1])
    output_data[0] = value


@avelang.jit
def kernel_invariant_raw_buffer_subview(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((1,), S.i32),
):
    invariant.tag(input_data, lambda i: (i,), "RawSubviewTag")
    view = S.subview(input_data, (2,), (4,), (1,))
    rsrc = S.amdgpu.make_rsrc(view, S.convert(16, S.i32))
    value = S.amdgpu.raw_buffer_load_x1(
        rsrc, S.convert(0, S.i32), S.convert(4, S.i32), 0
    )
    invariant.assert_tag_eq(value, input_data[3])
    output_data[0] = value


@avelang.jit
def kernel_invariant_tag_capture(output_data: S.Tensor((1,), S.i32)):
    base = S.thread_id(0)
    fragment = S.make_local((4,), S.i32)
    invariant.tag(fragment, lambda lane: (base + lane,), "CapturedTag", base)
    output_data[0] = fragment[0]


class TestInvariantTagSyntax(unittest.TestCase):
    def test_validation_mode_changes_the_normalized_jit_cache_key(self):
        compiler = AmdgpuCompiler(GPUTarget("amdgcn-amd-amdhsa", "gfx942"))
        original = os.environ.get("AVELANG_VALIDATE_INVARIANTS")
        try:
            os.environ["AVELANG_VALIDATE_INVARIANTS"] = "0"
            disabled = compiler.parse_options({"num_warps": 8})
            os.environ["AVELANG_VALIDATE_INVARIANTS"] = "1"
            enabled = compiler.parse_options({"num_warps": 8})
        finally:
            if original is None:
                os.environ.pop("AVELANG_VALIDATE_INVARIANTS", None)
            else:
                os.environ["AVELANG_VALIDATE_INVARIANTS"] = original

        cache = {}
        disabled_key = compute_cache_key(cache, [("*bf16", None)], disabled)
        enabled_key = compute_cache_key(cache, [("*bf16", None)], enabled)
        self.assertNotEqual(disabled_key, enabled_key)

    def test_shared_copy_example_lowers_tag_ops(self):
        mlir = generate_mlir(kernel_invariant_shared_copy)

        self.assertIn("ave.tag.bind %arg0", mlir)
        self.assertIn('tag_name = "LaneTag"', mlir)
        self.assertIn("ave.tag.assert_eq", mlir)
        self.assertIn("gpu.barrier", mlir)
        self.assertIn("ave.memref.load", mlir)
        self.assertIn("ave.memref.store", mlir)

    def test_reset_example_lowers_reset_op(self):
        mlir = generate_mlir(kernel_invariant_reset_shared)

        self.assertIn("ave.tag.bind %arg0", mlir)
        self.assertIn('tag_name = "LaneTag"', mlir)
        self.assertIn("ave.tag.reset", mlir)
        self.assertIn("#gpu.address_space<workgroup>", mlir)

    def test_loop_example_keeps_assertion_inside_loop(self):
        mlir = generate_mlir(kernel_invariant_loop_index)

        self.assertIn("ave.tag.bind %arg0", mlir)
        self.assertIn('tag_name = "IndexTag"', mlir)
        self.assertIn("scf.for", mlir)
        self.assertIn("ave.tag.assert_eq", mlir)
        self.assertIn("ave.memref.load %arg0[%arg2]", mlir)

    def test_z3_reports_unsat_for_generated_equivalent_invariants(self):
        mlir = generate_mlir(kernel_invariant_z3_equivalent)
        bindings = extract_tag_bindings(mlir)

        self.assertIn("BaseTag", bindings)
        self.assertIn("EquivalentTag", bindings)
        self.assertIn("ave.tag.assert_eq", mlir)

        lhs_input = bindings["BaseTag"]["inputs"][0]
        rhs_input = bindings["EquivalentTag"]["inputs"][0]
        lhs_expr = tag_expr_to_smt(bindings["BaseTag"]["exprs"][0])
        rhs_expr = tag_expr_to_smt(bindings["EquivalentTag"]["exprs"][0])

        smtlib = f"""
(set-logic QF_NIA)
(declare-const {lhs_input} Int)
(declare-const {rhs_input} Int)
(assert (= {lhs_input} {rhs_input}))
(assert (not (= {lhs_expr} {rhs_expr})))
(check-sat)
"""
        self.assertEqual(run_z3(smtlib), "unsat")

    def test_z3_reports_sat_for_generated_mismatched_invariants(self):
        mlir = generate_mlir(kernel_invariant_z3_mismatch)
        bindings = extract_tag_bindings(mlir)

        self.assertIn("BaseTag", bindings)
        self.assertIn("ShiftedTag", bindings)
        self.assertIn("ave.tag.assert_eq", mlir)

        lhs_input = bindings["BaseTag"]["inputs"][0]
        rhs_input = bindings["ShiftedTag"]["inputs"][0]
        lhs_expr = tag_expr_to_smt(bindings["BaseTag"]["exprs"][0])
        rhs_expr = tag_expr_to_smt(bindings["ShiftedTag"]["exprs"][0])

        smtlib = f"""
(set-logic QF_NIA)
(declare-const {lhs_input} Int)
(declare-const {rhs_input} Int)
(assert (= {lhs_input} {rhs_input}))
(assert (not (= {lhs_expr} {rhs_expr})))
(check-sat)
"""
        self.assertEqual(run_z3(smtlib), "sat")

    def test_raw_buffer_load_keeps_tag_assertion_and_descriptor(self):
        mlir = generate_mlir(kernel_invariant_raw_buffer_load)

        self.assertIn('tag_name = "RawBufferTag"', mlir)
        self.assertIn("ave.gpu.amdgpu_raw_buffer_load", mlir)
        self.assertIn("ave.tag.assert_eq", mlir)

    def test_raw_buffer_lds_keeps_tag_assertion_and_memory_path(self):
        mlir = generate_mlir(kernel_invariant_raw_buffer_load_lds)

        self.assertIn('tag_name = "RawBufferLdsTag"', mlir)
        self.assertIn("raw_buffer_load_lds_u32", mlir)
        self.assertIn("ave.memref.extract_aligned_pointer_as_index", mlir)
        self.assertIn("ave.tag.assert_eq", mlir)

    def test_raw_buffer_subview_keeps_root_tag_and_byte_offset(self):
        mlir = generate_mlir(kernel_invariant_raw_buffer_subview)

        self.assertIn('tag_name = "RawSubviewTag"', mlir)
        self.assertIn("ave.memref.subview", mlir)
        self.assertIn("ave.gpu.amdgpu_raw_buffer_load", mlir)
        self.assertIn("ave.tag.assert_eq", mlir)

    def test_tag_capture_keeps_ssa_operand_and_name(self):
        mlir = generate_mlir(kernel_invariant_tag_capture)

        self.assertIn('tag_name = "CapturedTag"', mlir)
        self.assertIn("tag_capture_names = [\"base\"]", mlir)


if __name__ == "__main__":
    unittest.main()
