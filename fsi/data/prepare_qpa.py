import argparse
import os
import pandas as pd
import pathlib

from typing import List, Optional, Tuple

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result-source-dir", type=pathlib.Path, help="Directory which contains raw result files."
)
parser.add_argument(
    "--chunk-source-dir", type=pathlib.Path, help="Directory which contains chunk files."
)
parser.add_argument("--output-dir", type=pathlib.Path, help="Directory to write output CSVs to.")
args = parser.parse_args()


def parse_good_question(text: str) -> Optional[str]:
    lower_text = text.lower()
    if lower_text.startswith("good question:"):
        return text[14:].strip()
    if lower_text.startswith("bad question:"):
        return None
    if lower_text.startswith("answer:"):
        return None
    return text


def parse_answer(text: str) -> Optional[str]:
    lower_text = text.lower()
    if "bad question:" in lower_text:
        return None
    if lower_text.startswith("answer:"):
        return text[7:].strip()
    return text


def parse_bad_question(text: str) -> Optional[str]:
    lower_text = text.lower()
    if lower_text.startswith("bad question:"):
        return text[13:].strip()
    else:
        return None


def try_parse_with_split(split_list: List[str]) -> Optional[Tuple[str, str, str]]:
    split_list = [x.strip() for x in split_list if x.strip() != ""]
    if len(split_list) not in [2, 3]:
        return None
    good = parse_good_question(split_list[0])
    answer = parse_answer(split_list[1])
    if len(split_list) == 3:
        bad = parse_bad_question(split_list[2])
    else:
        bad = ""
    if good is None or answer is None or bad is None:
        return None
    return (good, answer, bad)


def try_parse_with_indexing(text: str) -> Optional[Tuple[str, str, str]]:
    try:
        answer_idx = text.lower().index("answer:")
        good = text[:answer_idx].strip()
        if good.lower().startswith("good question:"):
            good = good[14:].strip()
        answer_and_bad = text[answer_idx + 7 :].strip()
        try:
            bad_question_idx = answer_and_bad.lower().index("bad question:")
            answer = answer_and_bad[:bad_question_idx].strip()
            bad = answer_and_bad[bad_question_idx + 13 :].strip()
        except:
            answer = answer_and_bad
            bad = ""
        return (good, answer, bad)
    except ValueError:
        return None


def split_text_into_good_answer_bad(text: str) -> Optional[Tuple[str, str, str]]:
    tup = try_parse_with_split(text.split("\n\n"))
    if tup is None:
        tup = try_parse_with_split(text.split("\n"))
    if tup is None:
        tup = try_parse_with_indexing(text)
    if tup is None:
        print(f"Unable to parse:\n{text}")
    return tup


base_path = args.result_source_dir
train_files = [os.path.join(base_path, x) for x in os.listdir(base_path) if x.startswith("train")]
test_files = [os.path.join(base_path, x) for x in os.listdir(base_path) if x.startswith("test")]
train_dfs = [pd.read_csv(x)[["chunk_id", "result"]] for x in train_files]
train_df = pd.concat(train_dfs).reset_index(drop=True)
test_dfs = [pd.read_csv(x)[["chunk_id", "result"]] for x in test_files]
test_df = pd.concat(test_dfs).reset_index(drop=True)

train_df[["good_question", "answer", "bad_question"]] = pd.DataFrame(
    train_df["result"].apply(lambda x: split_text_into_good_answer_bad(x)).tolist(),
    index=train_df.index,
)
train_df.dropna(inplace=True)
train_df = train_df[["chunk_id", "good_question", "answer", "bad_question"]]
test_df[["good_question", "answer", "bad_question"]] = pd.DataFrame(
    test_df["result"].apply(lambda x: split_text_into_good_answer_bad(x)).tolist(),
    index=test_df.index,
)
test_df.dropna(inplace=True)
test_df = test_df[["chunk_id", "good_question", "answer", "bad_question"]]

train_chunk_df = pd.read_csv(os.path.join(args.chunk_source_dir, "train_chunks.csv"))[
    ["chunk_id", "chunk"]
]
test_chunk_df = pd.read_csv(os.path.join(args.chunk_source_dir, "test_chunks.csv"))[
    ["chunk_id", "chunk"]
]
train_df = train_df.merge(train_chunk_df, on="chunk_id", how="left")
test_df = test_df.merge(test_chunk_df, on="chunk_id", how="left")

os.makedirs(args.output_dir, exist_ok=True)
train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
