import unittest
from pathlib import Path

import torch

from softmax_llir import _compile_llvm_ir, softmax_from_llir


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SoftmaxLlirTest(unittest.TestCase):
    def test_compiles_saved_ir_to_cuda_binary(self):
        ir = Path(__file__).with_name("softmax_kernel.llir").read_text()
        executable = _compile_llvm_ir(torch.cuda.current_device(), ir)
        self.assertTrue(executable.cubin.startswith(b"\x7fELF"))

    def test_matches_torch_for_multiple_shapes(self):
        torch.manual_seed(0)
        for rows, cols in [(1, 1), (7, 17), (32, 780), (1823, 781), (4, 1024), (4096, 8064), (4, 8192)]:
            with self.subTest(rows=rows, cols=cols):
                x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
                actual = softmax_from_llir(x)
                expected = torch.softmax(x, dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_contiguous_input_with_unaligned_storage_offset(self):
        x = torch.randn(2 * 8064 + 1, device="cuda")[1:].view(2, 8064)
        self.assertTrue(x.is_contiguous())
        actual = softmax_from_llir(x)
        torch.testing.assert_close(actual, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)

    def test_rejects_unsupported_inputs(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            softmax_from_llir(torch.randn(2, 3))
        with self.assertRaisesRegex(TypeError, "float32"):
            softmax_from_llir(torch.randn(2, 3, device="cuda", dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "8192"):
            softmax_from_llir(torch.randn(2, 8193, device="cuda"))


if __name__ == "__main__":
    unittest.main()
