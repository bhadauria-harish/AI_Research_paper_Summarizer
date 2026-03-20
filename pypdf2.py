import PyPDF2


def load_pdf(pdf_path: str) -> str:
    """
    Extract and return all text from a local PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full extracted text as a single string.
    """
    pages_text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)

if __name__ == "__main__":
    extracted_text = load_pdf("Image_Forgery_Detection.pdf")
    
    # Save the output to a text file
    with open("pypdf2.txt", "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)
        
    print("PDF text successfully extracted")