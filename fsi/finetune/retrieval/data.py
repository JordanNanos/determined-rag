import os
import glob 
import random
import datasets
from scipy.spatial.distance import cosine
from torch.utils.data import Dataset
from sentence_transformers import InputExample


class QPDataset(Dataset):
    def __init__(self, data_dir, embedding_func, split):
        """
        Load the dataset from the parsed data dir which should have train.csv and test.csv

        DatasetDict({
            train: Dataset({
                features: ['chunk_id', 'good_question', 'answer', 'bad_question', 'chunk'],
                num_rows: 7377
            })
            test: Dataset({
                features: ['chunk_id', 'good_question', 'answer', 'bad_question', 'chunk'],
                num_rows: 772
            })
        })
        """
        self.dataset = datasets.load_dataset(data_dir)
        self.embedding_func = embedding_func
        self.split = split
        #if split=="train":
        #    self.fill_missing_bad_questions()

    def fill_missing_bad_questions(self):
        # TODO: this doesn't work
        num_train = len(self.dataset["train"])
        self.good_qn_emb = [self.embedding_func(self.dataset["train"]["good_question"][i], is_query=True) for i in range(num_train)]
        for i in range(num_train):
            bad_question = self.dataset["train"]["bad_question"][i]
            if bad_question is None:
                bad_qn_emb = self.embedding_func(bad_question, is_query=True)
                highest_sim = -1
                best_match = 0
                for j in range(num_train):
                    if i!=j:
                        cosine_sim = 1 - cosine(bad_qn_emb[0], self.good_qn_emb[j])
                        if cosine_sim > highest_sim:
                            highest_sim = cosine_sim
                            best_match = j
            self.dataset["train"]["bad_question"][i] = self.dataset["train"]["good_question"][best_match]
        
    def __getitem__(self, item):
        sample = self.dataset[self.split][item]
        if self.split=="train":
            num_train = len(self.dataset["train"])
            bad_question = sample["bad_question"]
            if bad_question is None:
                sample["bad_question"] = self.dataset["train"][random.randrange(num_train)]["good_question"]
            return InputExample(
                texts=["{SOS}"+sample["chunk"], "[SOS]"+sample["good_question"], "[SOS]"+sample["bad_question"]]
            )
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset[self.split])

        
def create_corpus_from_file(file):
    dataset = datasets.load_dataset("csv", data_files=file)["train"]
    return {dataset[i]["chunk_id"]: dataset[i]["chunk"] for i in range(len(dataset))}
    
def create_corpus(chunk_dir):
    files = glob.glob(os.path.join(chunk_dir, "*chunks.csv"))
    corpus = {}
    for f in files:
        sub_corpus = create_corpus_from_file(f)
        corpus.update(sub_corpus)
    return corpus

def get_qa_eval_dict(data_dir, chunk_dir):
    dataset = datasets.load_dataset(data_dir)["test"]
    num_samples = len(dataset)
    queries = {dataset[i]["chunk_id"]: dataset[i]["good_question"] for i in range(num_samples)} # {qid: text}
    corpus = create_corpus(chunk_dir) # {pid: text}
    relevant_docs = {k: [k] for k in queries.keys()} # {qid: [pids]}
    return {
        "queries": queries,
        "corpus": corpus,
        "relevant_docs": relevant_docs
    }
