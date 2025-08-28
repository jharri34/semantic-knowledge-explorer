import os
import PyPDF2
import torch
import lancedb
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from typing import List, Dict, Any, Optional
import asyncio

import pyarrow as pa

# Initialize Rich Console for pretty output
console = Console()

# --- Configuration ---
DB_PATH = "./lancedb_data" # Local directory for LanceDB
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500 # Characters per chunk
CHUNK_OVERLAP = 50 # Overlap between chunks

# --- Global Resources ---
# These will be loaded once
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility Functions ---
def load_embedding_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        with console.status("[bold green]Loading embedding model...[/bold green]", spinner="dots"):
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
            console.log(f"[green]Embedding model '{EMBEDDING_MODEL_NAME}' loaded on {device}.[/green]")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        console.log(f"[green]Extracted text from {pdf_path}[/green]")
    except Exception as e:
        console.log(f"[red]Error extracting text from {pdf_path}: {e}[/red]")
    return text

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    console.log(f"[green]Chunked text into {len(chunks)} chunks.[/green]")
    return chunks

def generate_embedding(text: str) -> List[float]:
    """
    Generates a conceptual embedding for a given text using the loaded model.
    """
    if tokenizer is None or model is None:
        load_embedding_model() # Ensure model is loaded

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding (first token) as the sentence embedding
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()
    return embedding

async def ingest_document(pdf_path: str, db_table: lancedb.table.Table):
    """
    Ingests a PDF document: extracts text, chunks it, generates embeddings, and stores in LanceDB.
    """
    console.log(f"[bold blue]Ingesting document: {pdf_path}[/bold blue]")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        console.log(f"[red]No text extracted from {pdf_path}. Skipping ingestion.[/red]")
        return

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    data_to_insert = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        embedding_task = progress.add_task("[cyan]Generating embeddings...[/cyan]", total=len(chunks))
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            data_to_insert.append({
                "text": chunk,
                "vector": embedding,
                "source_document": os.path.basename(pdf_path),
                "chunk_id": i
            })
            progress.update(embedding_task, advance=1)

    if data_to_insert:
        with console.status("[bold green]Storing data in LanceDB...[/bold green]", spinner="dots"):
            db_table.add(data_to_insert)
            console.log(f"[green]Successfully ingested {len(data_to_insert)} chunks from {pdf_path}.[/green]")
    else:
        console.log(f"[yellow]No data to insert for {pdf_path}.[/yellow]")

async def semantic_search(query: str, db_table: lancedb.table.Table, top_k: int = 5):
    """
    Performs a semantic search in LanceDB for the given query.
    """
    console.log(f"[bold blue]Performing semantic search for: '{query}'[/bold blue]")
    query_embedding = generate_embedding(query)

    with console.status("[bold green]Searching LanceDB...[/bold green]", spinner="dots"):
        results = db_table.search(query_embedding).limit(top_k).to_list()
    
    if results:
        console.log(f"[bold green]Found {len(results)} nearest neighbors:[/bold green]")
        for i, result in enumerate(results):
            console.print(f"[bold yellow]Result {i+1}:[/bold yellow]")
            console.print(f"  [cyan]Source:[/cyan] {result['source_document']} (Chunk ID: {result['chunk_id']})")
            console.print(f"  [cyan]Similarity:[/cyan] {result['_distance']:.4f}") # LanceDB returns _distance
            console.print(f"  [cyan]Text:[/cyan] {result['text'][:200]}...") # Show first 200 chars
            console.print("\n")
    else:
        console.log("[yellow]No results found for your query.[/yellow]")

async def main():
    # Ensure embedding model is loaded before any operations
    load_embedding_model()

    # Connect to LanceDB
    db = lancedb.connect(DB_PATH)
    table_name = "conceptual_embeddings"

    # Define the schema using pyarrow
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)), # Vector field with dimension 384 for all-MiniLM-L6-v2
        pa.field("source_document", pa.string()),
        pa.field("chunk_id", pa.int32()),
    ])

    # Create or connect to table
    # Using mode="overwrite" for easy testing. For production, use "create_if_not_exists".
    db_table = db.create_table(table_name, schema=schema, mode="overwrite")
    console.log(f"[green]Table '{table_name}' created/connected (mode='overwrite').[/green]")

    # --- Example Usage ---
    # 1. Ingest a document (replace with your PDF path)
    example_pdf_path = "/mnt/d/Books/Yvonne P. Chireau - Black Magic_ Religion and the African American Conjuring Tradition (2003, University of California Press) - libgen.li.pdf" # <<< CHANGE THIS TO YOUR PDF PATH
    if os.path.exists(example_pdf_path):
        await ingest_document(example_pdf_path, db_table)
    else:
        console.log(f"[red]Example PDF not found at {example_pdf_path}. Please update the path or provide a PDF.[/red]")

    # 2. Perform a semantic search
    await semantic_search("What are the main themes of the book?", db_table)
    await semantic_search("Key concepts discussed in the text.", db_table)

if __name__ == "__main__":
    asyncio.run(main())
