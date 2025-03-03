import os
import random
import urllib.parse
from locust import HttpUser, task, between

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def download_image(self):
        # we generate a random image URL to simulate different images
        image_url = f"{API_URL}/{random.randint(1, 1000)}.jpg"
        params = {
            'image_url': image_url
        }
        self.client.get("/predict", params=params)

# Run the load test with:
# locust -f loadtest.py
# and visit http://localhost:8089 to start the test.
