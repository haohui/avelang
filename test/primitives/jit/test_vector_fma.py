#!/usr/bin/env python3
import unittest

import _avelang_bindings as _C
import avelang
import avelang.language as S
from avelang.compiler.code_generator import (
    _build_import_module,
    _collect_jit_dependencies,
    _get_function_def,
)


@avelang.jit
def scalar_fma(
    a: S.Tensor((1,), S.f32),
    b: S.Tensor((1,), S.f32),
    c: S.Tensor((1,), S.f32),
    out: S.Tensor((1,), S.f32),
):
    out[0] = S.fma(a[0], b[0], c[0])


@avelang.jit
def vector_bf16_fma(
    a: S.Tensor((2,), S.bf16),
    b: S.Tensor((2,), S.bf16),
    c: S.Tensor((2,), S.bf16),
    out: S.Tensor((2,), S.bf16),
):
    a_vec = S.view(a, S.Tensor((1, 2), S.bf16))[0]
    b_vec = S.view(b, S.Tensor((1, 2), S.bf16))[0]
    c_vec = S.view(c, S.Tensor((1, 2), S.bf16))[0]
    fused = S.fma(a_vec, b_vec, c_vec)
    separate = a_vec * b_vec + c_vec
    out[0] = fused[0]
    out[1] = separate[1]


@avelang.jit
def vector_f16_fma(
    a: S.Tensor((2,), S.f16),
    b: S.Tensor((2,), S.f16),
    c: S.Tensor((2,), S.f16),
    out: S.Tensor((2,), S.f16),
):
    a_vec = S.view(a, S.Tensor((1, 2), S.f16))[0]
    b_vec = S.view(b, S.Tensor((1, 2), S.f16))[0]
    c_vec = S.view(c, S.Tensor((1, 2), S.f16))[0]
    fused = S.fma(a_vec, b_vec, c_vec)
    separate = a_vec * b_vec + c_vec
    out[0] = fused[0]
    out[1] = separate[1]


@avelang.jit
def vector_f32_fma(
    a: S.Tensor((2,), S.f32),
    b: S.Tensor((2,), S.f32),
    c: S.Tensor((2,), S.f32),
    out: S.Tensor((2,), S.f32),
):
    a_vec = S.view(a, S.Tensor((1, 2), S.f32))[0]
    b_vec = S.view(b, S.Tensor((1, 2), S.f32))[0]
    c_vec = S.view(c, S.Tensor((1, 2), S.f32))[0]
    fused = S.fma(a_vec, b_vec, c_vec)
    separate = a_vec * b_vec + c_vec
    out[0] = fused[0]
    out[1] = separate[1]


def generate_mlir(jit_fn) -> str:
    jit_deps = _collect_jit_dependencies(jit_fn)
    import_module = _build_import_module([jit_fn, *jit_deps])
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)
    for dep in jit_deps:
        generator.visit_function_def(_get_function_def(dep.parse()), "[]", "jit")
    generator.visit_function_def(_get_function_def(jit_fn.parse()), "[]", "kernel")
    return generator.get_mlir()


class TestVectorFma(unittest.TestCase):
    def test_scalar_fma(self):
        self.assertIn("math.fma", generate_mlir(scalar_fma))

    def test_two_lane_floating_point_vectors(self):
        for jit_fn, element_type in (
            (vector_bf16_fma, "bf16"),
            (vector_f16_fma, "f16"),
            (vector_f32_fma, "f32"),
        ):
            with self.subTest(element_type=element_type):
                mlir = generate_mlir(jit_fn)
                vector_type = f"vector<2x{element_type}>"
                self.assertIn("vector.fma", mlir)
                self.assertIn("arith.mulf", mlir)
                self.assertIn("arith.addf", mlir)
                self.assertIn(vector_type, mlir)


if __name__ == "__main__":
    unittest.main()
