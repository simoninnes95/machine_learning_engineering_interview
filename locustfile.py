import os
from locust import HttpUser, task, between
import random

# Get the API host from the environment variable (default if not set)
API_HOST = os.getenv("API_HOST", "http://localhost:8000")

class RayServeUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_image_model(self):
        # Compose image URL from the API host and a sample image id
        random_image_id = random.randint(0, 100)
        image_url = f"{API_HOST}/{random_image_id}.jpg"
        # Send request to the ray serve endpoint with the image_url payload
        self.client.post("/", json={"image_url": image_url})
