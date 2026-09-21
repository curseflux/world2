import contextlib
import unittest
from unittest.mock import patch

import torch

from gridworld.runtime import device_and_dtype


class RuntimeTests(unittest.TestCase):
    def test_auto_and_bare_cuda_resolve_current_index(self):
        for requested in ('auto', 'cuda'):
            with self.subTest(device=requested), \
                 patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.current_device', return_value=2):
                device, dtype = device_and_dtype({'device': requested, 'precision': 'fp32'})
                self.assertEqual(device, torch.device('cuda:2'))
                self.assertEqual(dtype, torch.float32)
                # Reproduce the stricter contract of older mem_get_info versions.
                self.assertIsNotNone(device.index)

    def test_explicit_index_is_preserved_and_used_for_precision_detection(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.device', return_value=contextlib.nullcontext()) as context, \
             patch('torch.cuda.is_bf16_supported', return_value=True):
            device, dtype = device_and_dtype({'device': 'cuda:1', 'precision': 'auto'})
            self.assertEqual(device, torch.device('cuda:1'))
            self.assertEqual(dtype, torch.bfloat16)
            context.assert_called_once_with(torch.device('cuda:1'))

    def test_cpu_auto_does_not_query_cuda_device(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.cuda.current_device') as current:
            device, dtype = device_and_dtype({'device': 'auto', 'precision': 'auto'})
            self.assertEqual(device, torch.device('cpu'))
            self.assertEqual(dtype, torch.float32)
            current.assert_not_called()


if __name__ == '__main__':
    unittest.main()
