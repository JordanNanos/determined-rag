import torch


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

class SGPTEmbeddings:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        # Query and document prompt are ignored
        self.SPECB_QUE_TOK = tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self.SPECB_DOC_TOK = tokenizer.encode("{SOS}", add_special_tokens=False)[0]
    
    def encode(self, texts, is_query=True):
        features = []
        for text in texts:
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
            features.append(
                tokens_to_features(
                    {
                        "input_ids": torch.tensor([seq]).to(self.model.device),
                        "attention_mask": torch.tensor([att]).to(self.model.device),
                    },
                    self.model,
                )
            )
        return features
