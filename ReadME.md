# TutLM GraphRAG with DSPy

A sophisticated graph-based Retrieval-Augmented Generation (RAG) system built with DSPy and various LLM models for processing and analyzing documents.

## Features

- PDF and Markdown document processing
- Graph-based knowledge representation
- Multi-modal question answering capabilities
- Support for different types of queries (general, mathematical, code, visual)
- Parallel processing for efficient document handling
- Table of Contents (TOC) extraction
- Customizable embedding generation

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- dspy
- networkx
- numpy
- nltk
- rake-nltk
- pymupdf (fitz)
- pymupdf4llm
- ollama
- pandas
- tqdm
- pydantic

### Ollama Models Setup

The system requires the following Ollama models to be installed:
- mistral-nemo:latest (General purpose)
- mathstral:latest (Mathematical computations)
- llava:latest (Visual processing)
- deepseek-coder-v2:latest (Code-related queries)
- mxbai-embed-large (Embeddings)

## System Architecture

### Core Components

1. **Document Processing**
   - PDF and Markdown file reading
   - Table of Contents extraction
   - Parallel chapter processing

2. **Graph Construction**
   - Text segmentation (chapters, pages, paragraphs, sentences)
   - Embedding generation
   - Graph node and edge creation
   - Keyword extraction using RAKE

3. **Query Processing**
   - Sub-question generation
   - Relevant chunk retrieval
   - Context-aware answer generation

4. **Multi-Modal Support**
   - General text processing
   - Mathematical computations
   - Code analysis
   - Image understanding

## Usage

### Basic Usage

```python
from tools import load_graph
from QM import GraphRAG

# Load your graph
graph_path = "path/to/your/graph.gml"
G = load_graph(graph_path)

# Initialize GraphRAG
graph_rag = GraphRAG(graph=G)

# Ask a question
question = "Your question here"
answer = graph_rag.answer_query(query=question, mode="gen")
print(answer)
```

### Mode Selection

The system supports four different modes:
- `gen`: General text processing (default)
- `mat`: Mathematical computations
- `vis`: Visual processing (developer needed)
- `code`: Code-related queries

```python
# Example with different modes
math_answer = graph_rag.answer_query(query=question, mode="mat")
code_answer = graph_rag.answer_query(query=question, mode="code")
visual_answer = graph_rag.answer_query(query=question, mode="vis")
```

### Processing Documents

```python
from tools import process_pdfs_in_folder

# Process multiple PDFs
folder_path = "path/to/pdfs"
save_path = "path/to/save"
process_pdfs_in_folder(folder_path, save_path)
```

### Graph Operations

```python
from tools import save_graph, load_graph

# Save graph
save_graph(graph, "path/to/save/graph.gml")

# Load graph
loaded_graph = load_graph("path/to/graph.gml")
```

## File Structure

- `tools.py`: Core utilities and functions
- `Config.py`: Configuration and imports
- `QM.py`: Query processing and RAG implementation
- `ReadPDF.py`: PDF processing functionality
- `graphio.py`: Graph I/O operations

## Implementation Details

### Embedding Generation
The system uses the mxbai-embed-large model for generating embeddings:

```python
def get_embedding(text, model="mxbai-embed-large"):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
```

### Cosine Similarity Calculation
Relevance is determined using cosine similarity:

```python
def calculate_cosine_similarity(chunk, query_embedding, embedding):
    if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(embedding) == 0:
        return (chunk, 0)
    cosine_sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
    return (chunk, cosine_sim)
```

## Performance Optimization

The system implements several optimization techniques:
- Parallel processing for document handling
- Multi-threading for chapter processing
- Efficient graph storage and retrieval
- Caching of embeddings in graph nodes

## Limitations

- Requires significant computational resources for large documents
- Dependent on Ollama model availability
- Graph size can become large with extensive documents
- Processing time increases with document complexity

## Future Improvements

1. Enhanced caching mechanisms
2. Support for additional file formats
3. Improved parallel processing
4. Advanced context management
5. Extended multi-modal capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
