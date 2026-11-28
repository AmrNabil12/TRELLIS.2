import os
import unittest
from unittest.mock import patch

import torch

from runpod.server import _sampler_overrides, _stage_fingerprint, patch_cfg_sampler_batching


class SamplerStabilityTests(unittest.TestCase):
    def test_default_sampling_preserves_model_parameters(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_sampler_overrides(), ({}, {}))

    def test_worker_can_change_geometry_steps_and_texture_conditioning_independently(self):
        env = {
            "TRELLIS2_SHAPE_STEPS": "18",
            "TRELLIS2_TEXTURE_STEPS": "20",
            "TRELLIS2_TEXTURE_GUIDANCE": "1.5",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_sampler_overrides(), ({"steps": 18}, {"steps": 20, "guidance_strength": 1.5}))
        with patch.dict(os.environ, {"TRELLIS2_TEXTURE_GUIDANCE": "nan"}, clear=True):
            with self.assertRaisesRegex(ValueError, "between 1 and 10"):
                _sampler_overrides()

    def test_stage_fingerprint_compares_bfloat16_data_and_coordinate_order(self):
        feats = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
        self.assertEqual(_stage_fingerprint(feats, torch), _stage_fingerprint(feats.clone(), torch))
        self.assertNotEqual(_stage_fingerprint(feats, torch), _stage_fingerprint(feats.flip(0), torch))
        self.assertNotEqual(_stage_fingerprint(feats, torch), _stage_fingerprint(feats.float(), torch))

    def test_stable_sampling_disables_batched_guidance(self):
        class EulerSampler:
            def sample(self, *args, **kwargs):
                raise AssertionError("The test expects the streaming Euler loop")

            def _inference_model(self, model, state, t, cond, **kwargs):
                self.modes.append(self._batch_cfg_enabled)
                return torch.zeros_like(state)

        class GuidanceMixin:
            def _inference_model(self, *args, **kwargs):
                raise AssertionError("The test calls the Euler sampler directly")

        patch_cfg_sampler_batching(EulerSampler, GuidanceMixin, None, None, torch, batch_cfg=False)
        sampler = EulerSampler()
        sampler.modes = []
        sampler.sample(object(), torch.ones(1), steps=3, verbose=False)
        self.assertEqual(sampler.modes, [False, False, False])
