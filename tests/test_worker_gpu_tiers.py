import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from runpod.server import GenerationParams, Trellis2Engine, create_app


class WorkerGpuTierTests(unittest.TestCase):
    def test_nominal_gpu_capacity(self):
        engine = Trellis2Engine()
        engine.vram_gb = 23.52
        for pipeline_type in ("512", "1024", "1024_cascade"):
            engine.validate_pipeline_gpu(pipeline_type)
        with self.assertRaisesRegex(ValueError, "requires a 32 GB GPU"):
            engine.validate_pipeline_gpu("1536_cascade")

        engine.vram_gb = 31.85
        engine.validate_pipeline_gpu("1536_cascade")
        with self.assertRaisesRegex(ValueError, "requires a 24 GB GPU"):
            engine.validate_pipeline_gpu("1024_cascade")

    def test_wrong_gpu_rejected_at_submission_and_before_inference(self):
        engine = Trellis2Engine()
        engine.status = "ready"
        engine.vram_gb = 23.52
        engine._pipeline = object()
        engine._run_once = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            with TestClient(create_app(engine, jobs, api_key="", autoload=False)) as client:
                for path in ("/jobs", "/generate"):
                    response = client.post(
                        path,
                        data={"pipeline_type": "1536_cascade"},
                        files={"image": ("unused.png", b"unused", "image/png")},
                    )
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn("requires a 32 GB GPU", response.json()["detail"])
            with self.assertRaisesRegex(ValueError, "requires a 32 GB GPU"):
                engine.generate_glb(None, GenerationParams(pipeline_type="1536_cascade"), jobs / "mesh.glb")
        engine._run_once.assert_not_called()
