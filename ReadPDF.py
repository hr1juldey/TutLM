import fitz  # PyMuPDF
import pymupdf4llm as plm
from toc import extract_toc, get_chapter_page_ranges
from pydantic import BaseModel, ValidationError, model_validator
from typing import List, Optional
import concurrent.futures


# Define a Pydantic model for chapter entries
class ChapterEntry(BaseModel):
    title: str
    pages: List[int]


# Define a Pydantic model for readpdf parameters
class ReadPDFParams(BaseModel):
    filename: str
    savepath: Optional[str] = None  # This is optional but will be validated based on 'save'
    save: bool = False

    # Pydantic V2 model validator to ensure savepath is required if save is True
    @model_validator(mode='after')
    def validate_savepath(self):
        if self.save and not self.savepath:
            raise ValueError("savepath is required if save is True")
        return self


def extract_chapter_to_markdown(doc_path: str, entry: ChapterEntry) -> str:
    """
    Extracts content from a single chapter and converts it to Markdown.

    Parameters:
        doc_path (str): Path to the PDF document.
        entry (ChapterEntry): An instance of ChapterEntry containing chapter title and its page range.

    Returns:
        str: A string containing the chapter formatted in Markdown.
    """
    # Open the PDF document within each process to ensure each process has its own instance
    with fitz.open(doc_path) as doc:
        pages = entry.pages
        
        print(f"Reading chapter: {entry.title}")
        
        # Use pymupdf4llm to convert specified pages to Markdown
        md_text = plm.to_markdown(doc=doc, pages=pages, show_progress=False)  # Convert to Markdown
        
        return f"# {entry.title}\n\n{md_text}\n\n"  # Add chapter title as Markdown header


def readpdf(params: ReadPDFParams) -> str:
    """
    Reads a PDF file and extracts its content as Markdown.

    Parameters:
        params (ReadPDFParams): An instance of ReadPDFParams containing the filename and save option.

    Returns:
        str: The extracted Markdown content if save is False; otherwise, None.
    """
    
    # Extract TOC
    chapters = extract_toc(params.filename)

    # Get chapter names with their corresponding page ranges
    chapter_ranges = get_chapter_page_ranges(chapters)

    # Prepare ChapterEntry instances for each chapter
    chapter_entries = [ChapterEntry(title=entry['title'], pages=entry['pages']) for entry in chapter_ranges]

    # Use ProcessPoolExecutor to parallelize the extraction of chapters
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(extract_chapter_to_markdown, params.filename, entry): entry.title
            for entry in chapter_entries
        }

        full_markdown_content = ""
        for future in concurrent.futures.as_completed(futures):
            title = futures[future]
            try:
                markdown_content = future.result()
                full_markdown_content += markdown_content
            except Exception as e:
                print(f"Error processing chapter '{title}': {e}")

    if params.save:
        # Save the extracted Markdown content to a file if save is True
        output_path = params.savepath
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write(full_markdown_content)
            print(f"Markdown content extracted and saved to {output_path}")
        return None  # Return None since we've saved the output

    return full_markdown_content  # Return the Markdown content if not saving


# Function to be called from another file
def process_pdf(filename: str, save: bool = False, savepath: Optional[str] = None):
    try:
        # Prepare the parameters for the readpdf function
        params = ReadPDFParams(filename=filename, save=save, savepath=savepath)
        
        # Execute the PDF reading function
        if save:
            readpdf(params)
        else:
            markdown_output = readpdf(params)
            return markdown_output  # Return Markdown content if save is False

    except ValidationError as e:
        print(f"Validation error: {e}")
        return None


