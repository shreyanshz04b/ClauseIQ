import fitz
import pytesseract
import numpy as np
import cv2
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', '', text, flags=re.I)
    return text.strip()


def is_junk_page(text):
    t = text.lower()

    if len(t) < 80:
        return True

    junk_patterns = [
        "email", "@", "phone", "tel",
        "address", "+91"
    ]

    score = sum(1 for j in junk_patterns if j in t)

    return score >= 4


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 3
    )

    return thresh


def ocr_page(page):
    mat = fitz.Matrix(2.5, 2.5)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    processed = preprocess_image(img)

    text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6 -l eng+hin"
    )

    return text


def extract_blocks(page):
    blocks = page.get_text("blocks")

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    lines = []
    for b in blocks:
        txt = b[4].strip()

        if len(txt) < 5:
            continue

        lines.append(txt)

    return "\n".join(lines)


def extract_text_with_pages(file_path):
    results = []

    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)

        for i, page in enumerate(doc):
            text = extract_blocks(page)

            if not text or len(text.strip()) < 120:
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

    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(file_path)

        if img is None:
            return []

        processed = preprocess_image(img)

        text = pytesseract.image_to_string(
            processed,
            config="--oem 3 --psm 6 -l eng+hin"
        )

        text = clean_text(text)

        if text and not is_junk_page(text):
            results.append({
                "page": 1,
                "text": text
            })

    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = clean_text(f.read())

                if text and not is_junk_page(text):
                    results.append({
                        "page": 1,
                        "text": text
                    })
        except:
            return []

    return results