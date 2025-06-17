import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np


def process_vision_info(messages):
    """
    Process vision information from messages to extract images and videos.

    Args:
        messages (list): List of message dictionaries containing content with images/videos.

    Returns:
        tuple: (images, videos) where:
            - images: List of PIL.Image objects
            - videos: List of video data (currently None as video isn't implemented)
    """
    images = []
    videos = []

    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "image":
                        image_data = item["image"]
                        if isinstance(image_data, str):
                            # Handle image paths
                            if os.path.exists(image_data):
                                image = Image.open(image_data).convert("RGB")
                                images.append(image)
                            # Handle base64 encoded images
                            elif image_data.startswith("data:image/"):
                                encoding_prefix = "base64,"
                                image_b64_data = image_data.split(encoding_prefix)[1]
                                image_data = base64.b64decode(image_b64_data)
                                image = Image.open(BytesIO(image_data)).convert("RGB")
                                images.append(image)
                        elif isinstance(image_data, np.ndarray):
                            # Handle numpy array
                            image = Image.fromarray(image_data).convert("RGB")
                            images.append(image)
                        elif hasattr(image_data, "save"):
                            # Assume it's already a PIL.Image
                            images.append(image_data.convert("RGB"))
                    elif item["type"] == "video":
                        # This is a placeholder for video processing
                        # Actual implementation would depend on how videos are handled in the model
                        videos.append(None)

    return images, videos
