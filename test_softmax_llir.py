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
        for rows, cols in [(1, 1), (7, 17), (32, 780), (1823, 781), (4, 1024),
                           (4096, 8064), (4, 8192), (4096, 8320), (4, 16384)]:
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

    def test_reduction_boundaries_and_extreme_values(self):
        # Five rows exercises an incomplete four-warp CTA on the small path.
        torch.manual_seed(0)
        boundaries = [*range(256, 1025, 128), *range(1536, 16385, 512)]
        columns = sorted({16, *(n + delta for n in boundaries for delta in [-1, 0, 1]
                                if n + delta <= 16384)})
        for cols in columns:
            with self.subTest(cols=cols):
                x = torch.randn(5, cols, device='cuda') * 80
                x[0].fill_(10000)
                x[1].fill_(-10000)
                x[2].fill_(-float('inf'))
                x[2, -1] = 0
                x[3, 0] = float('inf')
                x[4, -1] = float('nan')
                torch.testing.assert_close(softmax_from_llir(x), torch.softmax(x, dim=1),
                                           rtol=1e-5, atol=1e-6, equal_nan=True)

    def test_empty_rows(self):
        x = torch.empty(0, 257, device='cuda')
        torch.testing.assert_close(softmax_from_llir(x), x)

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(5, 1537, device='cuda')
            actual = softmax_from_llir(x)
            expected = torch.softmax(x, dim=1)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two visible GPUs required')
    def test_input_on_noncurrent_device(self):
        original = torch.cuda.current_device()
        other = (original + 1) % torch.cuda.device_count()
        x = torch.randn(5, 2049, device=f'cuda:{other}')
        actual = softmax_from_llir(x)
        self.assertEqual(torch.cuda.current_device(), original)
        torch.testing.assert_close(actual, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)

    def test_rejects_unsupported_inputs(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            softmax_from_llir(torch.randn(2, 3))
        with self.assertRaisesRegex(TypeError, "float32"):
            softmax_from_llir(torch.randn(2, 3, device="cuda", dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "16384"):
            softmax_from_llir(torch.randn(2, 16385, device="cuda"))


if __name__ == "__main__":
    unittest.main()
