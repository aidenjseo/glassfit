"""Upload bytes -> BGR ndarray. cv2 is imported lazily (it ships with mediapipe)."""

import numpy as np

from glassfit.errors import InvalidImage


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/...) into an ``(H, W, 3)`` uint8 BGR array.

    Raises:
        InvalidImage: if ``data`` is empty or cannot be decoded.
    """
    if not data:
        raise InvalidImage("Empty image upload")

    import cv2  # lazy: transitively provided by mediapipe's opencv-contrib-python dependency

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage(
            "Could not decode image bytes",
            details={"bytes_received": len(data)},
        )
    return image
