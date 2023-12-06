# A domain-specific application for FSI Q&A

The app compares the output of a base model (no RAG) and a finetuned model (with RAG)

- `app`: The streamlit app for user to perform Q&A. Run from here.
- `build`: Files to build docker images to run the various components, if needed
- `data`: Data prep scripts, sample output, and launch command for weaviate db
- `finetune`: Determined experiments for fine-tuning the retrieval and chat models

Stack used:
- Determined AI - GPU scheduling, finetuning, data prep
- Weaviate - for storing the embeddings
- Langchain - for prompting
- Streamlit - for building the web interface
- OpenAssisstant - the chat model
- SGPT - the retrieval model
