import os
import random
import urllib.parse
from locust import HttpUser, task, between

API_URL = os.getenv("API_URL", " http://api-service.dill.svc.cluster.local:8080")

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def download_image(self):
        # we generate a random image URL to simulate different images
        image_url = f"{API_URL}/{random.randint(1, 1000)}.jpg"
        encoded_image_url = urllib.parse.quote(image_url, safe="")

        self.client.post("/predict", params={"image_url": encoded_image_url})

# Run the load test with:
# locust -f loadtest.py
# and visit http://localhost:8089 to start the test.
