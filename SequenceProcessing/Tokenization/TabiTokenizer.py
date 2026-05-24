"""
Tabbii tokenizer wrapper.

This class provides a unified BaseTokenizer-compatible interface
for the Tabbii Turkish tokenizer so it can be evaluated alongside
CosmosGPT2, TurkishTokenizer, and other baselines.

The tokenizer backend is loaded lazily on first use.
"""

from BaseTokenizer import BaseTokenizer


class TabbiiTokenizer(BaseTokenizer):
    """Wrapper for the Tabbii Turkish tokenizer."""

    DEFAULT_MODEL_PATH = "tabbii-tokenizer"

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
            Path or HuggingFace repo ID.

        add_special_tokens
            Whether to include BOS/EOS tokens.

        cache_dir, revision, local_files_only
            Forwarded to tokenizer loader.
        """

        self.model_name_or_path = (
            model_name_or_path or self.DEFAULT_MODEL_PATH
        )

        self.add_special_tokens = add_special_tokens

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
                    "TabbiiTokenizer requires transformers.\n"
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
    # BaseTokenizer API
    # ---------------------------------------------------------

    def encode(self, text: str) -> list:
        """Encode text into token IDs."""

        return self.tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
        )

    def decode(self, token_ids) -> str:
        """Decode token IDs back into text."""

        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=not self.add_special_tokens,
        )

    # ---------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list:
        """Return token strings."""

        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, token_ids) -> list:

        return self.tokenizer.convert_ids_to_tokens(
            list(token_ids)
        )

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

        loaded = (
            "loaded"
            if self._tokenizer is not None
            else "lazy"
        )

        return (
            f"TabbiiTokenizer("
            f"model={self.model_name_or_path!r}, "
            f"{loaded})"
        )


if __name__ == "__main__":

    sentence = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."

    tok = TabbiiTokenizer()

    print(f"Model: {tok.model_name_or_path}")
    print(f"Input: {sentence}")

    ids = tok.encode(sentence)

    print(f"IDs   ({len(ids)}): {ids}")

    print(f"Pieces: {tok.convert_ids_to_tokens(ids)}")

    print(f"Decoded: {tok.decode(ids)}")
