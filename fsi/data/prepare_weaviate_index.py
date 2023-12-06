import argparse
import os
import pathlib
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import transformers
import weaviate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer",
    type=str,
    default="Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    help="Huggingface tokenizer to use.",
)
parser.add_argument(
    "--embedding-model",
    type=str,
    default="Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    help="Model to use for embeddings.",
)
parser.add_argument(
    "--hf-cache-dir",
    type=str,
    default="/cstor/coreystaten/hf",
    help="Huggingface cache dir for tokenizer and model files.",
)
parser.add_argument(
    "--chunks-file",
    type=pathlib.Path,
    action="append",
    help="Directory which contains chunk files.",
)
parser.add_argument("--output", type=pathlib.Path, help="Path to persistence directory.")
parser.add_argument(
    "--prompt",
    type=str,
    default="{}",
    help="Prompt string to use for inserting chunks -- use {} where chunk is inserted.",
)
parser.add_argument(
    "--tokenize-logic",
    choices=["default", "sgpt"],
    default="default",
    help="Logic class to use for query/document tokenization.",
)
args = parser.parse_args()


def tokens_to_features(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(
            **batch_tokens, output_hidden_states=True, return_dict=True
        ).hidden_states[-1]

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
        .to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask
    return embeddings[0].tolist()


class HFEmbeddings:
    def __init__(self, tokenizer, model, query_prompt, document_prompt):
        self.tokenizer = tokenizer
        self.model = model
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        features = []
        for text in texts:
            batch_tokens = self.tokenizer(
                [self.document_prompt.format(text)],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            features.append(tokens_to_features(batch_tokens, self.model))
        return features

    def embed_query(self, text: str) -> List[float]:
        batch_tokens = self.tokenizer(
            [self.query_prompt.format(text)],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        return tokens_to_features(batch_tokens, self.model)


class SGPTEmbeddings:
    def __init__(self, tokenizer, model, query_prompt, document_prompt):
        self.tokenizer = tokenizer
        self.model = model
        # Query and document prompt are ignored
        self.SPECB_QUE_TOK = tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self.SPECB_DOC_TOK = tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(self.document_prompt.format(x), is_query=False) for x in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.encode([self.query_prompt.format(text)], is_query=True)[0]

    def encode(self, text, is_query=True):
        batch_tokens = self.tokenizer(
            [text],
            padding=False,
            truncation=True,
        )
        seq = batch_tokens["input_ids"][0]
        att = batch_tokens["attention_mask"][0]
        if is_query:
            seq.insert(0, self.SPECB_QUE_TOK)
        else:
            seq.insert(0, self.SPECB_DOC_TOK)
        att.insert(0, 1)
        return tokens_to_features(
            {
                "input_ids": torch.tensor([seq]).to(self.model.device),
                "attention_mask": torch.tensor([att]).to(self.model.device),
            },
            self.model,
        )


TOKENIZE_LOGIC_MAP = {"default": HFEmbeddings, "sgpt": SGPTEmbeddings}


def create_weaviate_database(client: weaviate.Client, chunks: List[Dict[str, Any]], embedder):
    client.schema.delete_all()
    client.schema.create_class(
        {
            "class": "Passage",
            "vectorizer": "none",
            "properties": [
                {"dataType": ["text"], "name": "chunk", "description": "Passage text"},
                {"dataType": ["int"], "name": "chunk_id", "description": "Unique passage ID"},
                {"dataType": ["text"], "name": "title", "description": "Article title"},
                {"dataType": ["text"], "name": "url", "description": "Article URL"},
            ],
        }
    )
    print("Creating embedding database.")
    with client.batch(batch_size=1) as batch:
        for chunk in chunks:
            obj = {
                "chunk_id": chunk["chunk_id"],
                "chunk": chunk["chunk"],
                "title": chunk["title"],
                "url": chunk["url"],
            }
            vector = embedder.embed_documents([chunk["chunk"]])[0]
            if np.isnan(vector).any():
                print("NaN")
                continue
            try:
                batch.add_data_object(obj, "Passage", vector=vector)
            except:
                print(f"SKIPPED {chunk['chunk_id']}")


def main() -> None:
    os.makedirs(args.output, exist_ok=True)
    client = weaviate.Client(
        embedded_options=weaviate.EmbeddedOptions(
            persistence_data_path=args.output,
            port=6682,
        )
    )
    print("Loading model for creating index")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.hf_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        cache_dir=args.hf_cache_dir,
    ).to("cuda:0")
    model.eval()
    chunks = pd.concat([pd.read_csv(x) for x in args.chunks_file]).fillna("")
    embeddings = TOKENIZE_LOGIC_MAP[args.tokenize_logic](tokenizer, model, args.prompt, args.prompt)
    print("Creating embedding index")
    start_time = time.time()
    create_weaviate_database(client, chunks.to_dict(orient="records"), embeddings)
    end_time = time.time()
    print(f"Finished creating embedding index, total time {end_time - start_time}")


with torch.inference_mode():
    main()
