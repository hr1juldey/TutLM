import fitz  # PyMuPDF
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# Step 1: Define a Pydantic model for TOC entries
class TOCEntry(BaseModel):
    level: int
    title: str
    page: int

def extract_toc(filename: str) -> List[TOCEntry]:
    """
    Extracts the table of contents from a PDF file.

    Parameters:
        filename (str): The path to the PDF file.

    Returns:
        List[TOCEntry]: A list of TOCEntry objects containing chapter names and their corresponding page numbers.
    """
    # Open the PDF document
    doc = fitz.open(filename)
    
    # Retrieve the table of contents
    toc = doc.get_toc(simple=True)
    
    # Prepare a list to store chapter names and page numbers
    chapters = []
    
    for entry in toc:
        level, title, page = entry  # Unpack TOC entry
        
        # Validate and create a TOCEntry instance
        try:
            toc_entry = TOCEntry(level=level, title=title, page=page)
            chapters.append(toc_entry)
        except ValidationError as e:
            print(f"Validation error for entry {entry}: {e}")
    
    return chapters

def get_chapter_page_ranges(toc_entries: List[TOCEntry]) -> List[Dict[str, Any]]:
    """
    Returns chapter names with their corresponding page ranges.

    Parameters:
        toc_entries (List[TOCEntry]): A list of TOCEntry objects.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with chapter titles and their page ranges.
    """
    result = []
    
    for i in range(len(toc_entries)):
        current_entry = toc_entries[i]
        
        # Determine the next entry's page number for range calculation
        if i + 1 < len(toc_entries):
            next_entry = toc_entries[i + 1]
            end_page = next_entry.page - 1  # End at the previous page of the next chapter
        else:
            end_page = -1  # Last chapter has no next entry
        
        result.append({
            'title': current_entry.title,
            'pages': list(range(current_entry.page, end_page + 1)) if end_page != -1 else [current_entry.page]
        })
    
    return result

# # Example usage
# filename = "/home/riju279/Documents/Code/DSPyG/Principles of Biochemistry Ft.Lehninger.pdf"
    
# # Extract TOC
# chapters = extract_toc(filename)

# # Get chapter names with their corresponding page ranges
# chapter_ranges = get_chapter_page_ranges(chapters)

# # Print chapter names and their corresponding page ranges
# for entry in chapter_ranges:
#     print(f"Chapter: {entry['title']}, Pages: {entry['pages']}")
    


