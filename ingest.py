import os
import glob
import logging
import pickle
from src.config import DATA_DIR, DB_DIR
from src.document_processor.parser import TeslaReportParser
from src.document_processor.chunker import SemanticChunker, extract_metadata_from_filename
from src.retriever.hybrid_search import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ingestion():
    cache_file = os.path.join(DB_DIR, "cached_documents.pkl")
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached documents from {cache_file}...")
        with open(cache_file, "rb") as f:
            all_documents = pickle.load(f)
        logger.info(f"Successfully loaded {len(all_documents)} documents from cache.")
    else:
        # Find all pdfs
        pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {DATA_DIR}. Please ensure reports are placed in this directory.")
            return
            
        chunker = SemanticChunker(max_chunk_size=1500)
        all_documents = []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process.")
        
        for file_path in pdf_files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing: {filename}")
            
            metadata_base = extract_metadata_from_filename(filename)
            
            parser = TeslaReportParser(file_path=file_path)
            elements = parser.parse()
            logger.info(f"Extracted {len(elements)} elements from {filename}")
            
            docs = chunker.chunk(elements, metadata_base)
            all_documents.extend(docs)
            logger.info(f"Created {len(docs)} chunks from {filename}")
            
        logger.info(f"Total documents to index: {len(all_documents)}")
        
        # Save cache
        logger.info(f"Saving documents to cache at {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(all_documents, f)
    
    # Needs embedding configuration in Environment variables (e.g. OPENAI_API_KEY)
    retriever = HybridRetriever()
    retriever.build_index(all_documents)
    
if __name__ == "__main__":
    # Ensure this runs in the right directory contexts
    # We can use os.environ to set dummy keys for testing or the real key
    run_ingestion()
