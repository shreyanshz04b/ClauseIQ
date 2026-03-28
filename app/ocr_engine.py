import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_pil_image(pil_img):
    img = np.array(pil_img)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return img


def ocr_image(pil_img):
    processed = preprocess_pil_image(pil_img)

    text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6 -l eng+hin"
    )

    return clean_text(text)


def extract_text(file_path):
    results = []

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=300)

        for i, page in enumerate(pages):
            text = ocr_image(page)

            if text:
                results.append({
                    "page": i + 1,
                    "text": text
                })

    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
        text = ocr_image(img)

        if text:
            results.append({
                "page": 1,
                "text": text
            })

    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = clean_text(f.read())

                if text:
                    results.append({
                        "page": 1,
                        "text": text
                    })
        except:
            return []

    return results