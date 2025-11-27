import os
import re
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from sklearn.feature_extraction.text import TfidfVectorizer
# import config

class DocumentExtractor:
    def __init__(self):
      
        artifacts_path = "model/docling"
        self.pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        # Initialize stop words once
        # self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=50  # Top keywords per document
            )
        # self.processed_docs = [] # This is no longer needed for keyword logic
    
    def _clean_text(self, text: str) -> str:
        """
        Removes HTML-style comments from the text using regex.
        This handles multi-line comments as well.
        """
        # The pattern looks for ''. re.DOTALL makes '.' match newlines as well.
        pattern = r''
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        # Also remove any potential leading/trailing whitespace left after cleaning
        return cleaned_text.strip()

    def extract(self, file_path: str) -> str:
        """Extracts text from a document, cleans it, and returns it as a markdown string."""
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        result = doc_converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        # Call the cleaning function before returning the text
        cleaned_markdown = self._clean_text(markdown_text)
        return cleaned_markdown
    
    def _get_keywords(self, text: str) -> list[str]:
        """Extracts top keywords from a single text chunk using TF-IDF."""
        
        # The TfidfVectorizer expects a list of documents. 
        # The 'text' (our chunk) is our single document.
        try:
            # We call fit_transform on *this chunk alone*.
            # This fixes the error by passing the whole chunk as one item in a list.
            tfidf_matrix = self.tfidf.fit_transform([text]) 
            
            feature_names = self.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top 10 keywords (as in your original logic)
            top_indices = scores.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

            # Just return the keywords for this chunk.
            return sorted(keywords)

        except ValueError as e:
            # This handles chunks that are empty or contain only stopwords
            if "empty vocabulary" in str(e):
                return []  # Return an empty list of keywords
            else:
                raise e # Re-raise any other unexpected ValueError

    def chunk_with_metadata(self, text: str, file_path: str) -> list[dict]:
        """
        Chunks the text by '## ' headings and embeds metadata 
        (filename, keywords) with each chunk.
        """
        filename = os.path.basename(file_path)
        splitter = ['## ']
        chunks_with_metadata = []
        
        current_chunk_lines = []

        for line in text.splitlines(keepends=True):
            # If we find a new heading, process the previous chunk
            if any(line.startswith(s) for s in splitter):
                if current_chunk_lines:
                    # Join lines to form the chunk text
                    chunk_text = ''.join(current_chunk_lines)
                    # Get keywords from this chunk
                    keywords = self._get_keywords(chunk_text)
                    
                    # Append the chunk data
                    chunks_with_metadata.append({
                        "text": chunk_text,
                        "metadata": {
                            "filename": filename,
                            "keywords": keywords
                        }
                    })
                    # Start a new chunk
                    current_chunk_lines = [line] # Keep the heading line for the new chunk
                else:
                    # This handles the case where the file starts with a heading
                    current_chunk_lines = [line]
            else:
                 current_chunk_lines.append(line)

        # Don't forget the last chunk!
        if current_chunk_lines:
            chunk_text = ''.join(current_chunk_lines)
            keywords = self._get_keywords(chunk_text)
            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "keywords": keywords
                }
            })
            
        return chunks_with_metadata

# if __name__ == "__main__":
#     doc_extractor = DocumentExtractor()
#     pdf_file_path = r"C:\Users\astmt\OneDrive\Desktop\LLM at the edge\PDfs\minutes_of_meeting_drones.pdf"
    
#     print(f"Extracting text from: {pdf_file_path}")
#     extracted_text = doc_extractor.extract(pdf_file_path)
    
#     print("Chunking text and extracting keywords...")
#     chunks = doc_extractor.chunk_with_metadata(extracted_text, pdf_file_path)
    
#     print(f"\nTotal chunks found: {len(chunks)}\n")
    
#     for i, chunk_data in enumerate(chunks):
#         print(f"--- Chunk {i+1} ---")
#         print(f"Filename: {chunk_data['metadata']['filename']}")
        
#         # Format keywords for clean printing
#         keywords_str = ", ".join(chunk_data['metadata']['keywords']) if chunk_data['metadata']['keywords'] else "None"
#         print(f"Keywords: {keywords_str}")
        
#         print("\nText:\n")
#         print(chunk_data['text'])
#         print("-" * 20 + "\n")