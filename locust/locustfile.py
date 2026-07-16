"""
Locust flood-request simulation for the pneumonia classifier API.

Run headless, e.g.:
    locust -f locust/locustfile.py --headless -u 100 -r 20 -t 60s \
           --host http://localhost:8080 --csv results/scale_3

Or open the Locust web UI:
    locust -f locust/locustfile.py --host http://localhost:8080
"""

from pathlib import Path

from locust import HttpUser, between, task

# one real chest X-ray from the test set, bundled next to this file
SAMPLE_IMAGE = Path(__file__).parent / "sample_xray.png"
IMAGE_BYTES = SAMPLE_IMAGE.read_bytes()


class ModelUser(HttpUser):
    """Simulated user hammering the model with predictions."""

    wait_time = between(0.5, 2.0)

    @task(8)
    def predict(self):
        self.client.post(
            "/predict",
            files={"file": ("sample_xray.png", IMAGE_BYTES, "image/png")},
        )

    @task(1)
    def status(self):
        self.client.get("/status")
