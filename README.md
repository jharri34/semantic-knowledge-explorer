# Semantic Knowledge Explorer

## A Tool for Conceptual Knowledge Synthesis and Semantic Exploration

---

### Vision & Philosophy: "Knowledge Hacking" and Changing the Way You Learn

The Semantic Knowledge Explorer is born from the idea of **"knowledge hacking"** – actively engaging with information to uncover deeper connections and accelerate understanding. It aims to fundamentally **change the way you learn** by moving beyond passive consumption to active synthesis.

Inspired by the stages of learning (Concrete Experience, Reflective Observation, Abstract Conceptualization, and Active Experimentation), this tool is designed to be a smart assistant for anyone grappling with large volumes of information, particularly post-graduate researchers. Imagine having a system that can tell you: "This point was mentioned here and here, and by the way, this other source might help too," effectively connecting the dots across your entire curated library.

This project is about building a system that helps you synthesize information, explore themes, and find conceptually linked ideas across diverse sources, fostering a more dynamic and insightful learning process.

---

### What It Does (Features)

The Semantic Knowledge Explorer allows you to:

*   **Ingest Diverse Data:** Start by inputting PDF books and other textual documents.
*   **Generate Conceptual Embeddings:** Utilize powerful Large Language Models (LLMs) and AI models (specifically HuggingFace Transformers) to convert your text into high-dimensional conceptual embeddings.
*   **Semantic Search:** Go beyond keyword matching. Query your knowledge base with concepts, ideas, or passages and retrieve conceptually "close" neighbors in the embedding space.
*   **Discover Relationships:** Find other concepts, ideas, quotes, or even entire books that are semantically related to your query, based on meaning or subject.
*   **Structured Knowledge Base:** Store your processed data and embeddings efficiently in LanceDB, a modern vector database.

Future iterations aim for:

*   **Clustering & Visualization:** Visualize clusters of related concepts, showing similarity relationships and thematic groupings.
*   **Knowledge Graph Integration (GraphRAG):** Explore building dynamic knowledge graphs to represent complex relationships between entities and ideas.
*   **Learning Stage Support:** Develop interfaces and AI prompts that specifically support the different stages of learning, guiding users through reflection, conceptualization, and application.

---

### Getting Started

Follow these steps to set up and run your Semantic Knowledge Explorer.

#### Prerequisites

*   **Python 3.9+**: Ensure you have a compatible Python version installed.
*   **Git**: For cloning the repository.

#### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jharri34/semantic-knowledge-explorer.git
    cd semantic-knowledge-explorer
    ```

2.  **Install Dependencies:**
    This project uses `pip` and `pyproject.toml` for dependency management. It will install `transformers`, `lancedb`, `PyPDF2`, `rich`, `torch`, and `sentence-transformers`.
    ```bash
    pip install -e .
    ```

#### Configuration

1.  **Provide a PDF Document:**
    *   Open `main.py` in your project directory.
    *   Locate the line:
        ```python
        example_pdf_path = "/mnt/d/workspace/example.pdf" # <<< CHANGE THIS TO YOUR PDF PATH
        ```
    *   **Update this path** to point to an actual PDF file on your system that you wish to ingest. For example, if your PDF is at `D:\Documents\MyBook.pdf` (on Windows), you would change it to `/mnt/d/Documents/MyBook.pdf` (in WSL/Linux).

#### Running the Application

1.  **Execute the Script:**
    ```bash
    python main.py
    ```

    The script will:
    *   Load the embedding model.
    *   Extract text from your specified PDF.
    *   Chunk the text.
    *   Generate embeddings for each chunk.
    *   Store the chunks and embeddings in a LanceDB table (created in `./lancedb_data`).
    *   Perform example semantic searches and print the nearest neighbors to your console.

---

### Usage

*   **Ingest More Documents:** To add more documents to your knowledge base, simply change the `example_pdf_path` in `main.py` to a new PDF file and run the script again. The new content will be added to your existing LanceDB table.
*   **Perform Custom Searches:** Modify the `await semantic_search(...)` calls in `main.py` with your own queries to explore the conceptual relationships within your ingested documents.

---

### Next Steps & Roadmap

For a detailed roadmap of future enhancements, including improved chunking, advanced visualization, and integration with knowledge graphs, please refer to the `NEXT_STEPS.md` file in this repository.

---

### Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

---

### License

This project is open-source and available under the MIT License. (You may choose a different license if preferred.)
