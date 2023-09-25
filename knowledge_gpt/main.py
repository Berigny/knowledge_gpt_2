

import docx
import PyPDF2
from typing import Union

class FileWrapper:
    def __init__(self, file):
        self.file = file

    def is_valid(self):
        return self.file.size > 0

    def get_text(self):
        raise NotImplementedError("Subclasses must implement this method")


class DocxFileWrapper(FileWrapper):
    def get_text(self):
        doc = docx.Document(self.file)
        return " ".join([p.text for p in doc.paragraphs])


class TxtFileWrapper(FileWrapper):
    def get_text(self):
        return self.file.read()


class PdfFileWrapper(FileWrapper):
    def get_text(self):
        pdf_reader = PyPDF2.PdfFileReader(self.file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()
        return text


def read_file(uploaded_file) -> Union[FileWrapper, None]:
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return DocxFileWrapper(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return TxtFileWrapper(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        return PdfFileWrapper(uploaded_file)
    else:
        # handle unsupported file type
        return None

# Now your existing code continues here...

uploaded_files = st.file_uploader(
    "Upload pdf, docx, or txt files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Scanned documents are not supported yet!",
)
