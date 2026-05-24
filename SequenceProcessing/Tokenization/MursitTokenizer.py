"""
BERT-style Masked Language Modeling tokenizer wrapper.

This wrapper provides a unified BaseTokenizer-compatible interface
while additionally supporting dynamic MLM masking using the
standard 80/10/10 masking strategy from the original BERT paper.

The tokenizer backend is loaded lazily on first use.
"""

import random
import torch

from BaseTokenizer import BaseTokenizer


class BERTMLMTokenizer(BaseTokenizer):
    """Wrapper for BERT-style MLM tokenization."""

    DEFAULT_MODEL_ID = "bert-base-uncased"

    def __init__(
        self,
        model_name_or_path: str = None,
        mlm_probability: float = 0.15,
        cache_dir: str = None,
        revision: str = None,
        local_files_only: bool = False,
        return_tensors: str = "pt",
        **from_pretrained_kwargs,
    ):
        """
        Parameters
        ----------
        model_name_or_path
            HuggingFace tokenizer path or repo ID.

        mlm_probability
            Probability of masking tokens.

        return_tensors
            Tensor format ("pt", "np", etc.).
        """

        self.model_name_or_path = (
            model_name_or_path or self.DEFAULT_MODEL_ID
        )

        self.mlm_probability = mlm_probability

        self.return_tensors = return_tensors

        self._cache_dir = cache_dir
        self._revision = revision
        self._local_files_only = local_files_only
        self._extra_kwargs = from_pretrained_kwargs

        self._tokenizer = None

    # ---------------------------------------------------------
    # Lazy loading
    # ---------------------------------------------------------

    @property
    def tokenizer(self):

        if self._tokenizer is None:

            try:
                from transformers import AutoTokenizer

            except ImportError as exc:
                raise ImportError(
                    "BERTMLMTokenizer requires transformers.\n"
                    "Install with: pip install transformers"
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self._cache_dir,
                revision=self._revision,
                local_files_only=self._local_files_only,
                **self._extra_kwargs,
            )

        return self._tokenizer

    # ---------------------------------------------------------
    # Basic tokenization
    # ---------------------------------------------------------

    def encode(self, text: str):

        return self.tokenizer.encode(text)

    def decode(self, token_ids):

        return self.tokenizer.decode(token_ids)

    def tokenize(self, text: str):

        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, token_ids):

        return self.tokenizer.convert_ids_to_tokens(
            list(token_ids)
        )

    # ---------------------------------------------------------
    # MLM processing
    # ---------------------------------------------------------

    def mlm_encode(self, text: str):
        """
        Apply BERT MLM masking.

        Returns
        -------
        dict
            {
                input_ids,
                labels,
                attention_mask
            }
        """

        inputs = self.tokenizer(
            text,
            return_tensors=self.return_tensors,
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        labels = input_ids.clone()

        # ---------------------------------------------
        # Ignore special tokens
        # ---------------------------------------------

        special_tokens_mask = (
            self.tokenizer.get_special_tokens_mask(
                input_ids,
                already_has_special_tokens=True,
            )
        )

        special_tokens_mask = torch.tensor(
            special_tokens_mask,
            dtype=torch.bool,
        )

        # ---------------------------------------------
        # Select masked positions
        # ---------------------------------------------

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability,
        )

        probability_matrix.masked_fill_(
            special_tokens_mask,
            value=0.0,
        )

        masked_indices = torch.bernoulli(
            probability_matrix
        ).bool()

        labels[~masked_indices] = -100

        # ---------------------------------------------
        # 80 / 10 / 10 masking strategy
        # ---------------------------------------------

        for i in range(len(input_ids)):

            if not masked_indices[i]:
                continue

            rand = random.random()

            # 80% replace with [MASK]
            if rand < 0.8:

                input_ids[i] = (
                    self.tokenizer.mask_token_id
                )

            # 10% random token
            elif rand < 0.9:

                input_ids[i] = random.randint(
                    0,
                    self.tokenizer.vocab_size - 1,
                )

            # 10% unchanged
            else:
                pass

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def vocab_size(self):

        return self.tokenizer.vocab_size

    @property
    def mask_token_id(self):

        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self):

        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):

        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self):

        return self.tokenizer.sep_token_id

    def __len__(self):

        return self.vocab_size

    def __repr__(self):

        loaded = (
            "loaded"
            if self._tokenizer is not None
            else "lazy"
        )

        return (
            f"BERTMLMTokenizer("
            f"model={self.model_name_or_path!r}, "
            f"{loaded})"
        )


if __name__ == "__main__":

    sentence = (
        "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
    )

    tok = BERTMLMTokenizer()

    batch = tok.mlm_encode(sentence)

    print("Masked Input:")
    print(
        tok.convert_ids_to_tokens(
            batch["input_ids"]
        )
    )

    print("\nTarget Tokens:")
    print(
        tok.convert_ids_to_tokens(
            batch["labels"][
                batch["labels"] != -100
            ]
        )
    )
