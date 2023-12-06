- `prepare_chunks.py`: Convert scraped full documents (output by Kannan's scripts) to chunk CSVs.
- `prepare_chroma_index.py`: Creates a Chroma index from chunks using an embedding model.
- `prepare_weaviate_index.py`: Creates a Weaviate index from chunks using an embedding model.
- `prepare_qpa.py`: Convert GPT-4 output question/results to a CSV format.

Example usages are in `run_commands.sh`.