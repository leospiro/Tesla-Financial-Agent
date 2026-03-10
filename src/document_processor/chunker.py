import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Common SEC section headers (regex variations)
SEC_SECTION_PATTERN = re.compile(
    r'^(?:PART [IV]+|Item\s+\w+\.)\s+.*', re.IGNORECASE
)

class SemanticChunker:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def _split_large_text(self, text: str, max_size: int) -> List[str]:
        """Simple fallback splitter for very large text blocks."""
        chunks = []
        # split by paragraphs roughly
        paragraphs = text.split('\n\n')
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = p
            else:
                current_chunk += "\n\n" + p
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def chunk(self, elements: List[Dict[str, Any]], base_metadata: Dict[str, Any]) -> List[Document]:
        """
        Groups parser elements into semantic chunks.
        Elements: [{'type': 'text'|'table', 'content': '...', 'page_num': 1}]
        
        Outputs Langchain Document format.
        """
        documents = []
        current_section = "General"
        text_buffer = []
        
        for element in elements:
            type_ = element.get('type')
            content = element.get('content', '').strip()
            page_num = element.get('page_num')
            
            if not content:
                continue
                
            if type_ == 'text':
                # Check if this text starts with a new section header
                # Sometimes a section header is the only thing on a line, 
                # or the first line of a paragraph.
                first_line = content.split('\n')[0].strip()
                if SEC_SECTION_PATTERN.match(first_line) and len(first_line) < 200:
                    # Flush existing buffer
                    if text_buffer:
                        full_text = "\n".join(text_buffer)
                        splits = self._split_large_text(full_text, self.max_chunk_size)
                        for split in splits:
                            meta = base_metadata.copy()
                            meta.update({'section': current_section, 'chunk_type': 'text', 'page': page_num})
                            documents.append(Document(page_content=split, metadata=meta))
                        text_buffer = []
                    
                    # Update current section
                    current_section = first_line
                    text_buffer.append(content)
                else:
                    text_buffer.append(content)
                    
            elif type_ == 'table':
                # For tables, we create a standalone chunk containing just the table
                # We prepend the current section to provide context
                meta = base_metadata.copy()
                meta.update({'section': current_section, 'chunk_type': 'table', 'page': page_num})
                
                # Contextualize the table
                table_content = f"Chapter Section: {current_section}\n\n{content}"
                documents.append(Document(page_content=table_content, metadata=meta))

        # Flush any remaining text
        if text_buffer:
            full_text = "\n".join(text_buffer)
            splits = self._split_large_text(full_text, self.max_chunk_size)
            for split in splits:
                meta = base_metadata.copy()
                meta.update({'section': current_section, 'chunk_type': 'text', 'page': page_num})
                documents.append(Document(page_content=split, metadata=meta))
                
        return documents

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extracts year, quarter, report_type from Tesla filenames like tsla-20220930-10Q.pdf
    """
    filename = filename.lower()
    meta = {
        'source': filename,
        'year': 0,
        'quarter': '',
        'report_type': ''
    }
    
    # Try to extract year from format like '20220930' or '-2022'
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        meta['year'] = int(year_match.group(1))
        
    if '10q' in filename or '10-q' in filename:
        meta['report_type'] = '10-Q'
        # infer quarter by month approx if format is yyyymmdd
        if '0331' in filename: meta['quarter'] = 'Q1'
        elif '0630' in filename: meta['quarter'] = 'Q2'
        elif '0930' in filename: meta['quarter'] = 'Q3'
    elif '10k' in filename or '10-k' in filename:
        meta['report_type'] = '10-K'
        meta['quarter'] = 'Q4'
        
    return meta
