import os
import textwrap
from typing import List

import langchain
import streamlit as st
import torch
import transformers
import threading
import weaviate

st.set_page_config(layout="wide", page_title="RAG Demo for Retail Q&A")

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# src - https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["db"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


MAX_TOKEN_LENGTH = 2048


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


class HFEmbeddings(langchain.embeddings.base.Embeddings):
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
        return self.encode(self.query_prompt.format(text), is_query=True)

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


######

# CSS for formatting top bar
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #7630EA;
        padding: 15px;
        color: white;
        margin-top: -82px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create top bar
st.markdown(
    """
    <div class="top-bar">
         <img src="https://www.xma.co.uk/wp-content/uploads/2018/06/hpe_pri_wht_rev_rgb.png" alt="HPE Logo" height="60">  
    </div>
    """,
    unsafe_allow_html=True,
)

######

st.title("RAG Demo for Retail Q&A")


@st.cache_resource
# base model
def load_embedder(
    embedding_model_name,
    embedding_tokenizer_name,
    tokenize_logic,
    cuda_device="cuda:0",
):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        embedding_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(cuda_device)
    model.eval()
    embeddings = TOKENIZE_LOGIC_MAP[tokenize_logic](tokenizer, model, "{}", "{}")
    return embeddings


INDEX = None
INDEX2 = None


def get_prompt():
    prompt_template = """
        <|prompter|>{question}<|endoftext|>
        <|assistant|>"""
    input_variables = ["question"]

    PROMPT = langchain.prompts.PromptTemplate(
        template=prompt_template, input_variables=input_variables
    )
    return PROMPT


def get_prompt_qa():
    prompt_template = (
        "Use the following pieces of context to answer the question at the end. "
        + "If you don't know the answer, just say that you don't know, don't try to make up an "
        + "answer."
        + """
    {context}

    <|prompter|>{question}<|endoftext|>
    <|assistant|>"""
    )
    input_variables = ["context", "question"]

    PROMPT = langchain.prompts.PromptTemplate(
        template=prompt_template, input_variables=input_variables
    )
    return PROMPT


@st.cache_resource
def load_chat_model(cuda_device="cuda:0"):
    import transformers

    chat_model_name = os.environ.get("CHAT_MODEL_NAME")
    chat_tokenizer_name = os.environ.get("CHAT_TOKENIZER_NAME")
    chat_tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_name)
    chat_model = transformers.AutoModelForCausalLM.from_pretrained(
        chat_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(cuda_device)
    chat_model.eval()
    chat_pipeline = transformers.pipeline(
        "text-generation",
        model=chat_model,
        tokenizer=chat_tokenizer,
        device=cuda_device,
        max_new_tokens=256,
    )
    llm = langchain.HuggingFacePipeline(pipeline=chat_pipeline)
    PROMPT = get_prompt()

    return langchain.chains.LLMChain(llm=llm, prompt=PROMPT)


def get_weaviate_retrievals(client: weaviate.Client, embedder: SGPTEmbeddings, query: str) -> List:
    ALPHA = 0.5
    vector = embedder.embed_query(query)
    x = (
        client.query.get("Passage", ["chunk", "chunk_id", "title", "url"])
        .with_hybrid(query=query, vector=vector, alpha=ALPHA)
        .with_limit(3)
        .do()
    )

    try:
        _ = x["data"]["Get"]["Passage"]
    except:
        x = (
            client.query.get("Passage", ["chunk", "chunk_id", "title", "url"])
            .with_near_vector({"vector": vector})
            .with_limit(5)
            .do()
        )
    return x["data"]["Get"]["Passage"]


class QAModel:
    def __init__(self, weaviate_client, embedder, chat_pipeline):
        self.weaviate_client = weaviate_client
        self.embedder = embedder
        self.chat_pipeline = chat_pipeline
        self.prompt = get_prompt_qa()

    def get_retrievals(self, query):
        return get_weaviate_retrievals(self.weaviate_client, self.embedder, query)

    def run(self, query):
        retrievals = self.get_retrievals(query)
        context = "\n\n".join(["Context: " + x["chunk"] for x in retrievals])
        full_prompt = self.prompt.format(context=context, question=query)
        return self.chat_pipeline(full_prompt)[0]["generated_text"][len(full_prompt) :]


@st.cache_resource
def load_qa_model(cuda_device):
    embedder = load_embedder(
        os.environ.get("EMBEDDING_FINE_TUNED_MODEL_NAME"),
        os.environ.get("EMBEDDING_FINE_TUNED_TOKENIZER_NAME"),
        os.environ.get("EMBEDDING_FINE_TUNED_TOKENIZE_LOGIC"),
        cuda_device="cuda:1",
    )

    # cache hack error from streamlit
    import transformers

    client = weaviate.Client(
        embedded_options=weaviate.EmbeddedOptions(
            persistence_data_path=os.environ.get("EMBEDDING_FINE_TUNED_INDEX_DIR"),
            port=int(os.environ.get("WEAVIATE_PORT")),
        )
    )
    chat_model_name = os.environ.get("CHAT_MODEL_NAME")
    chat_tokenizer_name = os.environ.get("CHAT_TOKENIZER_NAME")
    chat_tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_name)
    chat_model = transformers.AutoModelForCausalLM.from_pretrained(
        chat_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(cuda_device)
    chat_model.eval()
    chat_pipeline = transformers.pipeline(
        "text-generation",
        model=chat_model,
        tokenizer=chat_tokenizer,
        device=cuda_device,
        max_new_tokens=256,
    )
    return QAModel(client, embedder, chat_pipeline)


def run_inference(user_input, model, model_type, model_outputs):
    output = model.run(user_input)
    model_outputs[model_type] = output


if check_password():
    with torch.inference_mode():
        with st.spinner("Loading Q&A models..."):
            model_base = load_chat_model()
            model_fine_tuned = load_qa_model(cuda_device="cuda:1")

        example_input_questions_file_name = os.environ.get("EXAMPLE_INPUT_QUESTIONS_FILE_NAME")
        example_questions = []
        if example_input_questions_file_name:
            with open(example_input_questions_file_name) as f:
                example_questions = f.readlines()

        option = st.selectbox(
            "Sample Questions", example_questions, key="sample_questions_selectbox"
        )
        user_input = st.text_input(
            label="What's your question?",
            key="input",
            value=st.session_state.sample_questions_selectbox,
        )

        col1, col2 = st.columns(2)

        if user_input:
            model_outputs = {}
            threads = []

            ft_model_thread = threading.Thread(
                target=run_inference,
                args=(user_input, model_fine_tuned, "fine_tuned", model_outputs),
            )
            base_model_thread = threading.Thread(
                target=run_inference, args=(user_input, model_base, "base", model_outputs)
            )
            threads.append(ft_model_thread)
            threads.append(base_model_thread)
            with st.spinner("Processing input..."):
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            with col2:
                with st.spinner("Processing..."):
                    output = model_outputs["fine_tuned"]
                    st.markdown("### Finetuned Model, with RAG :smile: :smile: :smile:")
                    st.markdown(output)
                    st.divider()
                    st.markdown("#### Related Documents via embeddings from fine tuned model")
                    related_docs = model_fine_tuned.get_retrievals(user_input)
                    for i, x in enumerate(related_docs):
                        title = x["title"]
                        url = x["url"]
                        st.markdown(f'#{i+1}: from "{title}"\n\n[{url}]({url})\n')
                        st.text("\n".join(textwrap.wrap(x["chunk"], 80)))
                        if i < len(related_docs) - 1:
                            st.divider()
            with col1:
                with st.spinner("Processing..."):
                    output = model_outputs["base"]
                    st.markdown("### Base Model, no RAG")
                    st.markdown(output)
                    st.divider()
