"""
CosmosGPT2 tokenizer wrapper.

The "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
paper (Bayram et al., 2025) compares TurkishTokenizer against three
matched baselines, one of which is CosmosGPT2 — the BPE tokenizer that
ships with the Cosmos research group's Turkish GPT-2 models
(Kesgin et al., 2024, "Introducing cosmosGPT: Monolingual Training for
Turkish Language Models", arXiv:2404.17336).

This class is a thin :class:`BaseTokenizer` wrapper around the
HuggingFace ``transformers.AutoTokenizer`` so the rest of the project
can treat all four tokenizers uniformly. The underlying model is loaded
lazily on the first call to :meth:`encode` / :meth:`decode`, which keeps
``import`` cheap and lets callers run unit tests without needing
network access.

Example
-------
>>> tok = CosmosGPT2Tokenizer()
>>> ids = tok.encode("Türkiye Cumhuriyeti'nin başkenti Ankara'dır.")
>>> tok.decode(ids)
"Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
"""

from BaseTokenizer import BaseTokenizer


class CosmosGPT2Tokenizer(BaseTokenizer):
    """HuggingFace-backed wrapper for the CosmosGPT2 BPE tokenizer."""

    #: HuggingFace repository ID. Override via the constructor argument
    #: when pointing at a local snapshot or a different checkpoint.
    DEFAULT_MODEL_ID = "ytu-ce-cosmos/turkish-gpt2-medium"

    def __init__(
        self,
        model_name_or_path: str = None,
        add_special_tokens: bool = False,
        cache_dir: str = None,
        revision: str = None,
        local_files_only: bool = False,
        **from_pretrained_kwargs,
    ):
        """
        Parameters
        ----------
        model_name_or_path
            HuggingFace repo ID or path to a local tokenizer snapshot.
            Defaults to :attr:`DEFAULT_MODEL_ID`.
        add_special_tokens
            Whether ``encode`` should prepend/append the tokenizer's
            special tokens (BOS/EOS). The cosmosGPT2 paper trains its
            GPT-2 models without explicit BOS/EOS, so the default is
            False to keep encode/decode round-tripping cleanly.
        cache_dir, revision, local_files_only
            Forwarded to ``AutoTokenizer.from_pretrained``.
        from_pretrained_kwargs
            Any additional keyword arguments to forward to
            ``AutoTokenizer.from_pretrained`` (e.g.
            ``trust_remote_code=True``).
        """
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_ID
        self.add_special_tokens = add_special_tokens
        self._cache_dir = cache_dir
        self._revision = revision
        self._local_files_only = local_files_only
        self._extra_kwargs = from_pretrained_kwargs
        self._tokenizer = None  # lazy

    # load

    @property
    def tokenizer(self):
        """The lazily-instantiated HuggingFace tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "CosmosGPT2Tokenizer needs the 'transformers' package. "
                    "Install it with `pip install transformers`."
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self._cache_dir,
                revision=self._revision,
                local_files_only=self._local_files_only,
                **self._extra_kwargs,
            )
        return self._tokenizer

    # BaseTokenizer

    def encode(self, text: str) -> list:
        """Encode ``text`` to a list of integer token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

    def decode(self, token_ids) -> str:
        """Decode a list of integer token IDs back to a string."""
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=not self.add_special_tokens,
        )

    # convenience extras

    def tokenize(self, text: str) -> list:
        """Return the raw subword token strings (no ID conversion)."""
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, token_ids) -> list:
        return self.tokenizer.convert_ids_to_tokens(list(token_ids))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def __repr__(self):
        loaded = "loaded" if self._tokenizer is not None else "lazy"
        return f"CosmosGPT2Tokenizer(model={self.model_name_or_path!r}, {loaded})"


if __name__ == "__main__":  # small demo
    sentence = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
    tok = CosmosGPT2Tokenizer()
    print(f"Model: {tok.model_name_or_path}")
    print(f"Input: {sentence}")
    ids = tok.encode(sentence)
    print(f"IDs   ({len(ids)}): {ids}")
    print(f"Pieces: {tok.convert_ids_to_tokens(ids)}")
    print(f"Decoded: {tok.decode(ids)}")
