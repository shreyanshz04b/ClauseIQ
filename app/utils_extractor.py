import fitz
import pytesseract
import numpy as np
import cv2
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_junk_page(text):
    text = text.lower()

    junk_patterns = [
        "defence colony",
        "psl chambers",
        "email",
        "e:",
        "t:",
        "tel",
        "mumbai",
        "bengaluru",
        "chandigarh",
        "address",
        "india –",
        "@",
        "+91"
    ]

    score = sum(1 for j in junk_patterns if j in text)

    if score >= 3:
        return True

    if len(text.strip()) < 120:
        return True

    return False


def ocr_page(page):
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return pytesseract.image_to_string(thresh, config="--psm 6")


def extract_text_with_pages(file_path):
    results = []

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)

        for i, page in enumerate(doc):
            text = page.get_text()

            if not text or len(text.strip()) < 50:
                text = ocr_page(page)

            text = clean_text(text)

            if not text:
                continue

            if is_junk_page(text):
                continue

            results.append({
                "page": i + 1,
                "text": text
            })

    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(file_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        text = pytesseract.image_to_string(thresh, config="--psm 6")
        text = clean_text(text)

        if text and not is_junk_page(text):
            results.append({
                "page": 1,
                "text": text
            })

    elif file_path.endswith(".txt"):
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = clean_text(f.read())

            if text and not is_junk_page(text):
                results.append({
                    "page": 1,
                    "text": text
                })

    return results