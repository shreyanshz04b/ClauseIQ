import pytesseract
from PIL import Image
from pdf2image import convert_from_path

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        pages = convert_from_path(file_path)
        text = ""
        for p in pages:
            text += pytesseract.image_to_string(p)
        return text

    elif file_path.endswith((".png",".jpg",".jpeg")):
        return pytesseract.image_to_string(Image.open(file_path))

    return ""