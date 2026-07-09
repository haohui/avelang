import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


@avelang.jit
def kernel_atomic_add_i32(value: S.i32, out: S.Tensor((1,), S.i32)):
    offset = S.convert(0, S.i32)
    S.amdgpu.atomic_add(offset, value, out, 1)


@avelang.jit
def kernel_atomic_add_packed_bf16(
    value: S.Tensor((2,), S.bf16),
    out: S.Tensor((2,), S.bf16),
):
    offset = S.convert(0, S.i32)
    packed_value = S.view(value, S.Tensor((1, 2), S.bf16))[0]
    S.amdgpu.atomic_add(offset, packed_value, out, 1)


@unittest.skipUnless(has_rocm(), "Requires ROCm/HIP with an AMD GPU.")
class TestAMDGPUAtomicAdd(unittest.TestCase):
    def test_i32_atomic_add(self):
        out = torch.tensor([11], dtype=torch.int32, device="cuda")

        kernel_atomic_add_i32[lambda: ((1, 1, 1), (1, 1, 1))](7, out)

        self.assertEqual(out.cpu().item(), 18)

    def test_packed_bf16_atomic_add(self):
        value = torch.tensor([1.5, -2.0], dtype=torch.bfloat16, device="cuda")
        out = torch.tensor([3.0, 4.0], dtype=torch.bfloat16, device="cuda")

        kernel_atomic_add_packed_bf16[lambda: ((1, 1, 1), (1, 1, 1))](value, out)

        expected = torch.tensor([4.5, 2.0], dtype=torch.bfloat16, device="cuda")
        self.assertTrue(torch.equal(out, expected), f"Expected {expected}, got {out}")
