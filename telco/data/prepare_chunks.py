import ast
import argparse
import copy
import hashlib
import langchain
import math
import os
import pathlib
import pandas as pd
import pickle
from typing import Dict, Tuple
import transformers
import urllib.parse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer", type=str, help="Huggingface tokenizer to use for measuring chunk lengths."
)
parser.add_argument("--hf-cache-dir", type=str, help="Huggingface cache dir for tokenizer files.")
parser.add_argument(
    "--source-dir", type=pathlib.Path, help="Directory which contains parquet and pkl source files."
)
parser.add_argument("--output-dir", type=pathlib.Path, help="Directory to write output CSVs to.")
parser.add_argument
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_cache_dir)


def find_source_files(path: str):
    paths = []
    for filename in os.listdir(path):
        if filename.endswith(".pkl") or filename.endswith(".parquet"):
            paths.append(os.path.join(path, filename))
    return paths


# The following code was used to find non-ascii UTF-8 characters.
# import re
# uc_chars = set()
# for chunk in chunks:
#     uc_chars = uc_chars.union(set(re.sub('[ -~]', '', chunk)))
# Characters were examined to see which were logical spaces.
# This replacement ensures that CharacterTextSplitter splits chunks appropriately.


def normalize_spaces(chunk_entry: Dict) -> Dict:
    chunk = chunk_entry["content_text"]
    chunk = chunk.replace("\xa0", " ")
    chunk = chunk.replace("\u200a", " ")
    chunk = chunk.replace("\u200b", " ")
    chunk = chunk.replace("\u202f", " ")
    ret_val = copy.deepcopy(chunk_entry)
    ret_val["content_text"] = chunk
    return ret_val


def read_file(path: str) -> pd.DataFrame:
    """
    Reads file into a DataFrame with the following columns:
    - title
    - url
    - date
    - all_content
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            df = pickle.load(f)
    else:
        raise Exception("Unknown file type.")
    # Flow parquet files had all_content as a string representation of a dict instead of a dict.
    if type(df["all_content"].iloc[[0]].item()) is str:
        df["all_content"] = df["all_content"].map(lambda x: ast.literal_eval(x))
    # CTM files had all_content as a dict with a single key, also called 'all_content'
    else:
        df["all_content"] = df["all_content"].map(lambda x: x["all_content"])
    # Normalize spaces for text splitter.
    df["all_content"] = df["all_content"].map(lambda x: [normalize_spaces(y) for y in x])
    # Omit entries with no content.
    df = df[df["all_content"].map(lambda x: len(x) != 0)]
    df["source"] = df["url"].map(lambda x: urllib.parse.urlparse(x).netloc)
    return df[["source", "title", "url", "date", "all_content"]]


def keep_full_text_chunk(x: str) -> bool:
    """
    Whether to keep this chunk in the full text for training and display.
    """
    if len(x.strip()) == 0:
        return False
    if x.strip().startswith("Sources"):
        return False
    return True


def keep_index_chunk(x: str) -> bool:
    """
    Whether to keep this chunk in the index for retrieval.
    """
    if not (keep_full_text_chunk(x)):
        return False
    # Omitting chunks < 50 tokens to reduce noise -- most of these did not look useful to retrieve.
    if len(tokenizer(x)["input_ids"]) < 50:
        return False
    return True


def train_test_split(
    data: pd.DataFrame, test_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # No sort / randomization is performed here -- assuming presorted.
    test_idx = math.floor(len(data) * (1.0 - test_ratio))
    train = data[:test_idx]
    test = data[test_idx:]
    return train, test


def save_document_csv(path: str, df: pd.DataFrame) -> None:
    df = df[["source", "url", "title", "date", "full_text"]]
    df.to_csv(path)


def save_document_fulltext_only_csv(path: str, df: pd.DataFrame) -> None:
    df = df[["full_text"]]
    df.to_csv(path, index=False)


def to_chunk_df(df: pd.DataFrame, chunk_id_offset: int) -> pd.DataFrame:
    splitter = langchain.text_splitter.CharacterTextSplitter(
        ".",
        chunk_size=500,
        chunk_overlap=100,
        length_function=lambda x: len(tokenizer(x)["input_ids"]),
    )
    df = df.explode("all_content")
    df["header"] = df["all_content"].map(lambda x: x["header"])
    df["chunk"] = df["all_content"].map(lambda x: splitter.split_text(x["content_text"].strip()))
    df = df[df["chunk"].map(lambda x: len(x) > 0)]
    df = df.explode("chunk")
    df = df[df["chunk"].map(lambda x: keep_index_chunk(x))]
    df.reset_index(drop=False, inplace=True)
    df.index += chunk_id_offset
    df.index.name = "chunk_id"
    return df


def save_chunk_csv(path: str, df: pd.DataFrame) -> None:
    df = df[
        [
            "document_id",
            "source",
            "url",
            "title",
            "date",
            "header",
            "chunk",
        ]
    ]
    df.to_csv(path)


filepaths = find_source_files(args.source_dir)
dfs = [read_file(path) for path in filepaths]
df = pd.concat(dfs)

df["full_text"] = df["all_content"].map(
    lambda x: "\n".join(
        [y["content_text"] for y in x if keep_full_text_chunk(y["content_text"])]
    ).strip()
)
df["hash"] = df["full_text"].map(lambda x: hashlib.md5(x.encode()).hexdigest())
# Duplicates in CTM data
df.drop_duplicates("hash", inplace=True)
# Drop empty documents.
df = df[df["full_text"].map(lambda x: len(x) > 0)]
df.sort_values("hash", inplace=True)
df.drop("hash", axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.index.name = "document_id"
train_df, test_df = train_test_split(df)
train_chunk_df = to_chunk_df(train_df, chunk_id_offset=0)
test_chunk_df = to_chunk_df(test_df, chunk_id_offset=len(train_chunk_df))

save_document_csv(args.output_dir / "train_documents.csv", train_df)
save_document_csv(args.output_dir / "test_documents.csv", test_df)
# Fulltext only CSVs are the correct format for the finetuning script.
save_document_fulltext_only_csv(args.output_dir / "train_documents_fulltext_only.csv", train_df)
save_document_fulltext_only_csv(args.output_dir / "test_documents_fulltext_only.csv", test_df)
save_chunk_csv(args.output_dir / "train_chunks.csv", train_chunk_df)
save_chunk_csv(args.output_dir / "test_chunks.csv", test_chunk_df)
