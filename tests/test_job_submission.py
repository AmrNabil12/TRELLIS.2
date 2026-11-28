import io
import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from runpod.server import MATERIAL_AUDIT_NAME, PROJECTION_COMPARISON_NAME, create_app


class JobSubmissionTests(unittest.TestCase):
    def test_retry_uses_one_job_even_across_workers(self):
        generated = []
        finished = Event()
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(image, "png")

        def generate(_image, _params, path):
            generated.append(path.parent.name)
            path.write_bytes(b"test GLB")
            path.with_name(MATERIAL_AUDIT_NAME).write_bytes(b"test audit")
            path.with_name(PROJECTION_COMPARISON_NAME).write_bytes(b"test comparison")
            finished.set()

        engine = SimpleNamespace(status="ready", stage="ready", generate_glb=generate)
        header = {"Idempotency-Key": "0123456789abcdef" * 2}
        files = {"image": ("image.png", image.getvalue(), "image/png")}
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            with TestClient(create_app(engine, jobs, api_key="", autoload=False)) as first:
                original = first.post("/jobs", headers=header, files=files)
                self.assertEqual(original.status_code, 202, original.text)
                self.assertTrue(finished.wait(2))
            job = original.json()["job_id"]
            with TestClient(create_app(engine, jobs, api_key="", autoload=False)) as second:
                replay = second.post("/jobs", headers=header, files=files)
                self.assertEqual(replay.status_code, 202, replay.text)
                self.assertEqual(replay.json()["job_id"], job)
                conflict = second.post("/jobs", headers=header, files=files, data={"seed": 9})
                self.assertEqual(conflict.status_code, 409)
                details = second.get(f"/jobs/{job}").json()
                self.assertEqual(second.get(details["material_audit_url"]).content, b"test audit")
                self.assertEqual(second.get(details["projection_comparison_url"]).content, b"test comparison")
            self.assertEqual(generated, [job])
