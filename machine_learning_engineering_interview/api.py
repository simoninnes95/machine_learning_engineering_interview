from PIL import Image, ImageDraw
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import random

api = FastAPI()

@api.get(
    "/{image_id}.jpg",
    summary="Download simulation image",
    description="Generates an image for download.",
)
async def download_image(image_id: str):
    # Seed the random number generator for determinism
    seed = hash(image_id)
    random.seed(seed)

    # Create a new image
    img_size = (256, 256)
    img = Image.new("RGB", img_size, color="white")
    draw = ImageDraw.Draw(img)

    # Generate some random shapes
    for _ in range(10):
        shape_type = random.choice(["rectangle", "ellipse"])
        x0 = random.randint(1, img_size[0])
        y0 = random.randint(1, img_size[1])
        x1 = random.randint(1, img_size[0])
        y1 = random.randint(1, img_size[1])
        # Ensure x0 <= x1 and y0 <= y1
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        color = tuple(random.randint(0, 255) for _ in range(3))

        if shape_type == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)

    # Save the image to a buffer
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(buf, media_type="image/jpeg")