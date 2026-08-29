"""
Secure Image Upload Validator
Validates image headers via magic bytes, enforces payload size bounds,
verifies MIME formats, and detects corrupted image streams.
"""

import io
import logging
from PIL import Image
from backend.app.core.exceptions import ImageProcessingError

logger = logging.getLogger("dairy_ai.core.file_validator")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Magic byte signatures for supported image types
MAGIC_BYTES = {
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"]
}


def validate_image_file(file_bytes: bytes, filename: str = "upload.jpg") -> Image.Image:
    """
    Validates uploaded image byte stream:
    1. Checks for non-empty content.
    2. Enforces maximum 10MB payload size limit.
    3. Inspects magic-byte file signatures (JPEG, PNG, WebP).
    4. Decodes image with PIL to verify integrity.
    5. Returns PIL Image object in RGB format.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ImageProcessingError(f"Uploaded file '{filename}' is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ImageProcessingError(
            f"Uploaded file '{filename}' exceeds maximum allowed size of 10 MB "
            f"({len(file_bytes) / (1024 * 1024):.2f} MB)."
        )

    # Magic byte signature verification
    header = file_bytes[:16]
    is_valid_signature = False

    # Check JPEG
    if header.startswith(b"\xFF\xD8\xFF"):
        is_valid_signature = True
    # Check PNG
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        is_valid_signature = True
    # Check WebP ("RIFF" at offset 0, "WEBP" at offset 8)
    elif header.startswith(b"RIFF") and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
        is_valid_signature = True

    if not is_valid_signature:
        logger.warning(f"File security rejection: '{filename}' failed magic byte signature check.")
        raise ImageProcessingError(
            f"File '{filename}' failed security check. File header does not match valid JPEG, PNG, or WebP magic bytes."
        )

    # PIL decode verification
    try:
        buf = io.BytesIO(file_bytes)
        img = Image.open(buf)
        img.verify()  # Verify structural integrity
        
        # Re-open stream for actual RGB conversion since verify() mutates file position
        buf.seek(0)
        img_rgb = Image.open(buf).convert("RGB")
        return img_rgb
    except Exception as e:
        logger.error(f"PIL decode failure for '{filename}': {e}")
        raise ImageProcessingError(f"File '{filename}' is corrupted or unreadable as a valid image: {str(e)}")
