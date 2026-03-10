import pdfplumber
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def is_within_bboxes(obj, bboxes):
    """Check if a pdfplumber object is within any of the given bounding boxes."""
    # obj has x0, top, x1, bottom
    for bbox in bboxes:
        bx0, btop, bx1, bbottom = bbox
        if (obj['x0'] >= bx0 and obj['x1'] <= bx1 and 
            obj['top'] >= btop and obj['bottom'] <= bbottom):
            return True
    return False

def table_to_markdown(table_data: List[List[str]]) -> str:
    """Convert a 2D list of strings (from pdf table) to a markdown table."""
    if not table_data or len(table_data) == 0:
        return ""
    
    # Process headers
    headers = table_data[0]
    # Clean None values in headers
    headers = [str(h).replace('\n', ' ') if h is not None else "" for h in headers]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|-" + "-|-".join(["-" * max(1, len(h)) for h in headers]) + "-|\n"
    
    # Process rows
    for row in table_data[1:]:
        cleaned_row = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
        md_table += "| " + " | ".join(cleaned_row) + " |\n"
        
    return md_table

def unmerge_table_cells(table_data: List[List[Any]]) -> List[List[str]]:
    """ Basic cleaning up for table extraction to handle merged cells or newlines. """
    cleaned_data = []
    for row in table_data:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).strip())
        cleaned_data.append(cleaned_row)
    return cleaned_data

class TeslaReportParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parses the PDF and returns a list of elements (texts and text-tables) page by page.
        Returns:
            List format: [{'type': 'text'|'table', 'content': str, 'page_num': int}]
        """
        elements = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # 1. Find and extract tables
                    tables = page.find_tables()
                    table_bboxes = [t.bbox for t in tables]
                    
                    for table in tables:
                        table_data = table.extract()
                        if table_data:
                            cleaned_data = unmerge_table_cells(table_data)
                            md_table = table_to_markdown(cleaned_data)
                            elements.append({
                                'type': 'table',
                                'content': md_table,
                                'page_num': page_num
                            })
                            
                    # 2. Extract text outside tables to avoid duplication
                    def not_in_table(obj):
                        try:
                            # only check objects with bounding boxes
                            if 'x0' in obj and 'top' in obj:
                                # Provide some margin around table bbox to avoid clipping text tightly
                                for bbox in table_bboxes:
                                    bx0, btop, bx1, bbottom = bbox
                                    if (obj['x0'] >= bx0 - 2 and obj['x1'] <= bx1 + 2 and 
                                        obj['top'] >= btop - 2 and obj['bottom'] <= bbottom + 2):
                                        return False
                        except Exception:
                            pass
                        return True
                    
                    # Filter out the text that is strictly inside tables
                    filtered_page = page.filter(not_in_table)
                    text = filtered_page.extract_text()
                    if text:
                        elements.append({
                            'type': 'text',
                            'content': text.strip(),
                            'page_num': page_num
                        })
                        
        except Exception as e:
            logger.error(f"Failed to parse {self.file_path}: {e}")
            
        return elements

if __name__ == "__main__":
    # Test stub
    pass
