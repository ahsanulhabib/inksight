import os
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from PIL import Image


def process_document(file_path: str) -> Dict[str, Any]:
    """
    Processes a single document, extracting text and identifying its type.
    For images, it confirms the format and returns the path.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    try:
        if extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            # Verify it's a valid image
            with Image.open(file_path) as img:
                img.verify()
            return {"type": "image", "content": file_path, "error": None}

        elements = partition(filename=file_path)
        text_content = "\n".join([str(el) for el in elements])

        if not text_content.strip():
            return {"type": "text", "content": "", "error": "No text content found."}

        return {"type": "text", "content": text_content, "error": None}

    except Exception as e:
        return {
            "type": "error",
            "content": None,
            "error": f"Failed to process {os.path.basename(file_path)}: {e}",
        }
