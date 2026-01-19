# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from src.models.frame_decoder import FrameDecoderConfig, FrameDecoderViT


class TestFrameDecoder(unittest.TestCase):
    def test_decoder_output_shape(self):
        cfg = FrameDecoderConfig(img_size=256, patch_size=16, in_dim=64, embed_dim=64, depth=2, num_heads=4)
        dec = FrameDecoderViT(cfg)
        b = 3
        n = (256 // 16) * (256 // 16)
        tokens = torch.randn(b, n, 64)
        out = dec(tokens)
        self.assertEqual(tuple(out.shape), (b, 3, 256, 256))

    def test_decoder_raises_on_wrong_patch_count(self):
        cfg = FrameDecoderConfig(img_size=256, patch_size=16, in_dim=32, embed_dim=32, depth=1, num_heads=4)
        dec = FrameDecoderViT(cfg)
        tokens = torch.randn(2, 10, 32)
        with self.assertRaises(ValueError):
            _ = dec(tokens)

