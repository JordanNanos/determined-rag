"""
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)
"""
import argparse
import logging
import random
import os
from datetime import datetime

import numpy as np
import torch.cuda
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    models,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
import determined as det

from data import QPDataset, get_qa_eval_dict
from embeddings import SGPTEmbeddings


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--steps_per_epoch", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used? Multiple systems seperated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--train_dataset_max_size", default=None, type=int)
parser.add_argument("--dev_corpus_max_size", default=-1, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--model_save_path", default=None, type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument(
    "--add_special_token",
    action="store_true",
    help="Special tokens used by OpenAI with lasttoken pooling",
)
parser.add_argument("--speca", action="store_true")
parser.add_argument("--specb", action="store_true")
parser.add_argument("--asym", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument(
    "--freezenonbias",
    action="store_true",
    help="Freeze all except biases in transformer",
)
parser.add_argument(
    "--unfreezewte", action="store_true", help="Unfreeze Word Token Embeddings"
)
parser.add_argument("--gradcache", action="store_true")
parser.add_argument(
    "--chunksize", default=1, type=int, help="Chunks to use for gradcache"
)
parser.add_argument(
    "--data_dir", default=None, type=str, help="Directory for ques & passage data."
)
parser.add_argument(
    "--chunk_dir", default=None, type=str, help="Directory for corpus chunks."
)
args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = (
    args.train_batch_size
)  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = (
    args.max_seq_length
)  # Max length for passages. Increasing it, requires more GPU memory
num_epochs = args.epochs  # Number of epochs we want to train

dist = det.core.DistributedContext.from_torch_distributed()
with det.core.init(distributed=dist) as core_context:
    accelerator = Accelerator()

    # Load our embedding model
    if args.use_pre_trained_model:
        logging.info("use pretrained SBERT model")
        word_embedding_model = models.Transformer(
            model_name, max_seq_length=max_seq_length
        )
        if "gpt" in model_name.lower():
            word_embedding_model.tokenizer.pad_token = (
                word_embedding_model.tokenizer.eos_token
            )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), args.pooling
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        model.max_seq_length = max_seq_length
        print("loaded pretrained model")
    elif args.asym:
        logging.info("Create new asymmetric SBERT model")
        w1 = models.Transformer(model_name, max_seq_length=max_seq_length)
        w2 = models.Transformer(model_name, max_seq_length=max_seq_length)
        if args.add_special_token or args.speca:
            if args.add_special_token:
                tokens = ["[DOC]", "[QRY]"]
            elif args.speca:
                tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
                w1.bos_spec_token = w1.tokenizer.encode(
                    "[SOS]", add_special_tokens=False
                )
                w1.eos_spec_token = w1.tokenizer.encode(
                    "[EOS]", add_special_tokens=False
                )
                w2.bos_spec_token = w2.tokenizer.encode(
                    "[SOS]", add_special_tokens=False
                )
                w2.eos_spec_token = w2.tokenizer.encode(
                    "[EOS]", add_special_tokens=False
                )
            w1.tokenizer.add_tokens(tokens, special_tokens=True)
            w2.tokenizer.add_tokens(tokens, special_tokens=True)
            w1.auto_model.resize_token_embeddings(len(w1.tokenizer))
            w2.auto_model.resize_token_embeddings(len(w2.tokenizer))
        if "gpt" in model_name:
            w1.tokenizer.pad_token = w1.tokenizer.eos_token
            w2.tokenizer.pad_token = w2.tokenizer.eos_token
        assert w1.get_word_embedding_dimension() == w2.get_word_embedding_dimension()
        # Pooling has no weights, hence can be shared
        pooling = models.Pooling(w1.get_word_embedding_dimension(), args.pooling)

        asym_model = models.Asym(
            {"QRY": [w1], "DOCPOS": [w2], "DOCNEG": [w2]}, allow_empty_key=False
        )
        model = SentenceTransformer(modules=[asym_model, pooling])
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(
            model_name, max_seq_length=max_seq_length
        )
        if "gpt" in model_name.lower():
            word_embedding_model.tokenizer.pad_token = (
                word_embedding_model.tokenizer.eos_token
            )

        if args.add_special_token or args.speca:
            if args.add_special_token:
                tokens = ["[DOC]", "[QRY]"]
            elif args.speca:
                tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
            word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(
                len(word_embedding_model.tokenizer)
            )

            if args.speca:
                word_embedding_model.bos_spec_token_q = (
                    word_embedding_model.tokenizer.encode(
                        "[SOS]", add_special_tokens=False
                    )[0]
                )
                word_embedding_model.eos_spec_token_q = (
                    word_embedding_model.tokenizer.encode(
                        "[EOS]", add_special_tokens=False
                    )[0]
                )

                word_embedding_model.bos_spec_token_d = (
                    word_embedding_model.tokenizer.encode(
                        "{SOS}", add_special_tokens=False
                    )[0]
                )
                word_embedding_model.eos_spec_token_d = (
                    word_embedding_model.tokenizer.encode(
                        "{EOS}", add_special_tokens=False
                    )[0]
                )

        elif args.specb:
            tokens = ["[SOS]", "{SOS}"]
            word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(
                len(word_embedding_model.tokenizer)
            )

            # Will be replaced with the rep tokens in the model ones
            # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module,
            # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
            # If we would directly use the brackets here, they may become part of another token
            word_embedding_model.bos_spec_token_q = (
                word_embedding_model.tokenizer.encode(
                    "[SOS]", add_special_tokens=False
                )[0]
            )
            word_embedding_model.bos_spec_token_d = (
                word_embedding_model.tokenizer.encode(
                    "{SOS}", add_special_tokens=False
                )[0]
            )

            word_embedding_model.bos_spec_token_q_rep = (
                word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
            )
            word_embedding_model.eos_spec_token_q = (
                word_embedding_model.tokenizer.encode("]", add_special_tokens=False)[0]
            )

            word_embedding_model.bos_spec_token_d_rep = (
                word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
            )
            word_embedding_model.eos_spec_token_d = (
                word_embedding_model.tokenizer.encode("}", add_special_tokens=False)[0]
            )

            word_embedding_model.replace_bos = True

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), args.pooling
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if args.freeze or args.freezenonbias:
        for name, param in model.named_parameters():
            if args.freezenonbias and "bias" in name:
                # Freeze all except bias
                continue
            if args.unfreezewte and "wte" in name:
                # Do not freeze Word Token Embeddings
                continue
            param.requires_grad = False

    if args.model_save_path is None:
        model_save_path = "output/train_bi-encoder-mnrl-{}-{}".format(
            model_name.replace("/", "-"),
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    else:
        model_save_path = args.model_save_path

    # Create dataset
    data_folder = args.data_dir
    embedding_func = SGPTEmbeddings(
        word_embedding_model.tokenizer, word_embedding_model.auto_model
    ).encode

    if not args.no_training:
        train_dataset = QPDataset(data_folder, embedding_func, "train")

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=train_batch_size
        )
        if args.gradcache:
            train_loss = losses.MNRLGradCache(model, chunk_size=args.chunksize)
        else:
            train_loss = losses.MultipleNegativesRankingLoss(model)

        # Always take 1 ckpt per epoch - If 1 device -> 1 ckpt after whole trainloader
        # If X devices; Train-loader is X times bigger than actual steps
        checkpoint_save_steps = len(train_dataloader) // accelerator.num_processes
        logging.info(
            f"Dataloader length: {len(train_dataloader)}, CKPT Save Steps: {checkpoint_save_steps}"
        )

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=args.warmup_steps,
            use_amp=args.use_amp,
            checkpoint_path=model_save_path,
            checkpoint_save_steps=checkpoint_save_steps,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
            steps_per_epoch=args.steps_per_epoch,
            accelerator=accelerator,
            det_context=core_context,
            use_gradcache=args.gradcache,
            chunk_size=args.chunksize,
        )

        # Save the model
        model.save(model_save_path)

    # Evaluate
    ### Load eval data
    model = SentenceTransformer(model_save_path)

    if args.add_special_token or args.speca:
        word_embedding_model = model._first_module()
        assert isinstance(word_embedding_model, models.Transformer)

        if args.add_special_token:
            tokens = ["[DOC]", "[QRY]"]
        elif args.speca:
            tokens = ["[SOS]", "[EOS]", "{SOS}", "{EOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(
            len(word_embedding_model.tokenizer)
        )

        if args.speca:
            word_embedding_model.bos_spec_token_q = (
                word_embedding_model.tokenizer.encode(
                    "[SOS]", add_special_tokens=False
                )[0]
            )
            word_embedding_model.eos_spec_token_q = (
                word_embedding_model.tokenizer.encode(
                    "[EOS]", add_special_tokens=False
                )[0]
            )

            word_embedding_model.bos_spec_token_d = (
                word_embedding_model.tokenizer.encode(
                    "{SOS}", add_special_tokens=False
                )[0]
            )
            word_embedding_model.eos_spec_token_d = (
                word_embedding_model.tokenizer.encode(
                    "{EOS}", add_special_tokens=False
                )[0]
            )

    elif args.specb:
        word_embedding_model = model._first_module()
        assert isinstance(word_embedding_model, models.Transformer)

        tokens = ["[SOS]", "{SOS}"]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(
            len(word_embedding_model.tokenizer)
        )

        # Will be replaced with the rep ones
        word_embedding_model.bos_spec_token_q = word_embedding_model.tokenizer.encode(
            "[SOS]", add_special_tokens=False
        )[0]
        word_embedding_model.bos_spec_token_d = word_embedding_model.tokenizer.encode(
            "{SOS}", add_special_tokens=False
        )[0]

        word_embedding_model.bos_spec_token_q_rep = (
            word_embedding_model.tokenizer.encode("[", add_special_tokens=False)[0]
        )
        word_embedding_model.eos_spec_token_q = word_embedding_model.tokenizer.encode(
            "]", add_special_tokens=False
        )[0]

        word_embedding_model.bos_spec_token_d_rep = (
            word_embedding_model.tokenizer.encode("{", add_special_tokens=False)[0]
        )
        word_embedding_model.eos_spec_token_d = word_embedding_model.tokenizer.encode(
            "}", add_special_tokens=False
        )[0]

        word_embedding_model.replace_bos = True

    # only performing evaluation from one process
    eval_data = get_qa_eval_dict(args.data_dir, args.chunk_dir)

    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        eval_data["queries"],
        eval_data["corpus"],
        eval_data["relevant_docs"],
        show_progress_bar=True,
        corpus_chunk_size=100000,
        precision_recall_at_k=[10],
        batch_size=args.eval_batch_size,
        name="qa_test",
    )
    ir_evaluator(model)
