import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def aggregate_and_chunk_text(content):
    """Use Langchain splitter to chunk content for LLMs."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(content)

def classify_and_aggregate(content):
    """Classify markdown content and aggregate normal text into paragraphs."""
    
    # Define regex patterns
    page_break_pattern = r'(?:---|----)'
    heading_pattern = r'(#{1,7})\s*(.*)'
    bold_italic_pattern = r'(\*\*\*|___)(.*?)\1'
    bold_pattern = r'(\*\*|__)(.*?)\1'
    italic_pattern = r'(\*|_)(.*?)\1'
    inline_code_pattern = r'`([^`]*)`'
    code_block_pattern = r'```(.*?)```'
    ordered_list_pattern = r'^\d+\.\s+(.+)'
    unordered_list_pattern = r'^[-\*\+]\s+(.+)'
    table_pattern = r'^\|(.+)\|'
    link_pattern = r'\[(.*?)\]\((.*?)\)'
    emoji_pattern = r'(:[a-zA-Z0-9_+-]+:)'
    highlight_pattern = r'(==)(.*?)(==)'
    subscript_pattern = r'~(.*?)~'
    superscript_pattern = r'\^(.*?)\^'
    
    classifications = {
        'Page Number': [],
        'Index': [],
        'Type': [],
        'Content': []
    }
    
    # Split content by page breaks
    pages = re.split(page_break_pattern, content)

    global_index = 0  # To keep track of the original order across all pages

    for page_number, page_content in enumerate(pages, start=1):
        lines = page_content.split('\n')
        normal_text_accum = []  # Accumulate normal text to aggregate into paragraphs
        
        for line in lines:
            # If the line is a page break, classify it as "Page Break"
            if re.match(page_break_pattern, line.strip()):
                classifications['Type'].append('Page Break')
                classifications['Content'].append(line.strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                continue
            
            # Handle headings
            heading_match = re.match(heading_pattern, line)
            if heading_match:
                if normal_text_accum:
                    # If there is accumulated normal text, add it as a paragraph before the heading
                    classifications['Type'].append('Paragraph')
                    classifications['Content'].append(' '.join(normal_text_accum).strip())
                    classifications['Page Number'].append(page_number)
                    classifications['Index'].append(global_index)
                    global_index += 1
                    normal_text_accum = []
                # Add heading
                level = len(heading_match.group(1))
                classifications['Type'].append(f'Heading Level {level}')
                classifications['Content'].append(heading_match.group(2).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                continue
            
            # Classify other types
            match_found = False
            
            # Bold-italic
            bold_italic_match = re.search(bold_italic_pattern, line)
            if bold_italic_match:
                classifications['Type'].append('Bold Italic Text')
                classifications['Content'].append(bold_italic_match.group(2).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True
            
            # Bold
            bold_match = re.search(bold_pattern, line)
            if bold_match:
                classifications['Type'].append('Bold Text')
                classifications['Content'].append(bold_match.group(2).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True
            
            # Italic
            italic_match = re.search(italic_pattern, line)
            if italic_match:
                classifications['Type'].append('Italic Text')
                classifications['Content'].append(italic_match.group(2).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True
            
            # Inline code
            inline_code_match = re.search(inline_code_pattern, line)
            if inline_code_match:
                classifications['Type'].append('Inline Code')
                classifications['Content'].append(inline_code_match.group(1).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True
            
            # Code block
            code_block_match = re.search(code_block_pattern, line, re.DOTALL)
            if code_block_match:
                if normal_text_accum:
                    # Add aggregated normal text as a paragraph
                    classifications['Type'].append('Paragraph')
                    classifications['Content'].append(' '.join(normal_text_accum).strip())
                    classifications['Page Number'].append(page_number)
                    classifications['Index'].append(global_index)
                    global_index += 1
                    normal_text_accum = []
                classifications['Type'].append('Code Block')
                classifications['Content'].append(code_block_match.group(1).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True
            
            # Ordered and unordered lists
            ordered_list_match = re.match(ordered_list_pattern, line)
            unordered_list_match = re.match(unordered_list_pattern, line)
            if ordered_list_match or unordered_list_match:
                if normal_text_accum:
                    # Add aggregated normal text as a paragraph
                    classifications['Type'].append('Paragraph')
                    classifications['Content'].append(' '.join(normal_text_accum).strip())
                    classifications['Page Number'].append(page_number)
                    classifications['Index'].append(global_index)
                    global_index += 1
                    normal_text_accum = []
                list_type = 'Ordered List' if ordered_list_match else 'Unordered List'
                list_content = ordered_list_match.group(1).strip() if ordered_list_match else unordered_list_match.group(1).strip()
                classifications['Type'].append(list_type)
                classifications['Content'].append(list_content)
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Table
            table_match = re.match(table_pattern, line)
            if table_match:
                classifications['Type'].append('Table')
                classifications['Content'].append(table_match.group(1).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Link
            link_match = re.search(link_pattern, line)
            if link_match:
                classifications['Type'].append('Link')
                classifications['Content'].append(f"{link_match.group(1)}|{link_match.group(2)}")
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Emoji
            emoji_match = re.search(emoji_pattern, line)
            if emoji_match:
                classifications['Type'].append('Emoji')
                classifications['Content'].append(emoji_match.group(1))
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Highlight
            highlight_match = re.search(highlight_pattern, line)
            if highlight_match:
                classifications['Type'].append('Highlighted Text')
                classifications['Content'].append(highlight_match.group(2).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Subscript
            subscript_match = re.search(subscript_pattern, line)
            if subscript_match:
                classifications['Type'].append('Subscript')
                classifications['Content'].append(subscript_match.group(1).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Superscript
            superscript_match = re.search(superscript_pattern, line)
            if superscript_match:
                classifications['Type'].append('Superscript')
                classifications['Content'].append(superscript_match.group(1).strip())
                classifications['Page Number'].append(page_number)
                classifications['Index'].append(global_index)
                global_index += 1
                match_found = True

            # Classify normal text
            if not match_found and line.strip():
                normal_text_accum.append(line.strip())

        # If any normal text remains at the end of the page, add it as a paragraph
        if normal_text_accum:
            classifications['Type'].append('Paragraph')
            classifications['Content'].append(' '.join(normal_text_accum).strip())
            classifications['Page Number'].append(page_number)
            classifications['Index'].append(global_index)
            global_index += 1
    
    # Create the DataFrame
    markdown_df = pd.DataFrame(classifications)
    
    # Drop rows with empty content
    markdown_df = markdown_df[markdown_df['Content'].str.strip() != '']
    
    # Sort by page number and then by index
    markdown_df = markdown_df.sort_values(by=['Page Number', 'Index'])
    
    # Set 'Index' as the DataFrame index and drop the column
    markdown_df.set_index('Index', inplace=True)
    
    return markdown_df

def classify_markdown_file(file_path):
    """Classify markdown content from a file, aggregate normal text, and chunk if necessary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        classified_df = classify_and_aggregate(content)
        
        # Optionally, chunk the content for LLMs
        for idx, row in classified_df.iterrows():
            if row['Type'] == 'Paragraph':
                row['Content'] = aggregate_and_chunk_text(row['Content'])  # Use Langchain for chunking if needed
        
        return classified_df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
