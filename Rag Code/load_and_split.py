import os
import warnings
from pypdf import PdfReader

warnings.filterwarnings("ignore")


def load_all_pdfs_from_folder(folder_path: str) -> str:
    if not os.path.isdir(folder_path):
        raise ValueError("Provided path is not a valid folder")

    all_text = ""
    pdf_found = False

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_found = True
            reader = PdfReader(os.path.join(folder_path, filename))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

    if not pdf_found:
        raise ValueError("No PDF files found")

    if not all_text.strip():
        raise ValueError("PDFs contain no extractable text")

    return all_text


def split_text(text: str, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

