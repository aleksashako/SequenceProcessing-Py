"""
Comparison test: runs all four tokenizers on the same Turkish sentence
and prints encode / decode results side-by-side.

TurkishTokenizer runs fully offline (dummy morphological resources).
The three HuggingFace-backed tokenizers (CosmosGPT2, Mursit, TabiBERT)
require a network connection on first use to download model files.
If a download fails the test prints a warning and continues.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from SequenceProcessing.Tokenization.MorphologicalAnalyzer import MorphologicalAnalyzer
from SequenceProcessing.Tokenization.TurkishTokenizer import TurkishTokenizer
from SequenceProcessing.Tokenization.CosmosGPT2Tokenizer import CosmosGPT2Tokenizer
from SequenceProcessing.Tokenization.MursitTokenizer import MursitTokenizer
from SequenceProcessing.Tokenization.TabiTokenizer import TabiTokenizer

SENTENCE = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."

SEPARATOR = "=" * 60


class DummyBPE:
    """BPE fallback for TurkishTokenizer: returns the word as-is."""
    def encode(self, text: str):
        return [text]


def run_turkish():
    morph = MorphologicalAnalyzer(roots_path="dummy", affixes_path="dummy")
    tok = TurkishTokenizer(morph_analyzer=morph, bpe_fallback=DummyBPE())
    ids = tok.encode(SENTENCE)
    decoded = tok.decode(ids)
    return ids, decoded


def run_hf(tok_class):
    tok = tok_class()
    ids = tok.encode(SENTENCE)
    decoded = tok.decode(ids)
    pieces = tok.convert_ids_to_tokens(ids)
    return ids, decoded, pieces


def print_result(name, ids, decoded, pieces=None):
    print(f"\n{'─' * 60}")
    print(f"  Tokenizer : {name}")
    print(f"  Input     : {SENTENCE}")
    print(f"  # tokens  : {len(ids)}")
    if pieces:
        print(f"  Pieces    : {pieces}")
    print(f"  IDs       : {ids}")
    print(f"  Decoded   : {decoded}")
    ok = SENTENCE.replace(" ", "").lower() == decoded.replace(" ", "").lower()
    print(f"  Round-trip: {'✓ OK' if ok else '✗ MISMATCH'}")


def main():
    print(SEPARATOR)
    print("  ALL-TOKENIZER COMPARISON TEST")
    print(f"  Sentence: {SENTENCE}")
    print(SEPARATOR)

    # ── 1. TurkishTokenizer (offline, dummy resources) ──────────────────
    print("\n[1/4] TurkishTokenizer (morphological + BPE fallback)")
    try:
        ids, decoded = run_turkish()
        print_result("TurkishTokenizer", ids, decoded)
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ── 2. CosmosGPT2Tokenizer ───────────────────────────────────────────
    print("\n[2/4] CosmosGPT2Tokenizer  (ytu-ce-cosmos/turkish-gpt2-medium)")
    try:
        ids, decoded, pieces = run_hf(CosmosGPT2Tokenizer)
        print_result("CosmosGPT2Tokenizer", ids, decoded, pieces)
    except Exception as exc:
        print(f"  SKIPPED — could not load model: {exc}")

    # ── 3. MursitTokenizer ───────────────────────────────────────────────
    print("\n[3/4] MursitTokenizer  (newmindai/Mursit-Base)")
    try:
        ids, decoded, pieces = run_hf(MursitTokenizer)
        print_result("MursitTokenizer", ids, decoded, pieces)

        # MLM masking demo
        tok = MursitTokenizer()
        batch = tok.mlm_encode(SENTENCE)
        masked_pieces = tok.convert_ids_to_tokens(batch["input_ids"])
        target_pieces = tok.convert_ids_to_tokens(
            batch["labels"][batch["labels"] != -100]
        )
        print(f"  MLM masked : {masked_pieces}")
        print(f"  MLM targets: {target_pieces}")
    except Exception as exc:
        print(f"  SKIPPED — could not load model: {exc}")

    # ── 4. TabiTokenizer ─────────────────────────────────────────────────
    print("\n[4/4] TabiTokenizer  (boun-tabilab/TabiBERT)")
    try:
        ids, decoded, pieces = run_hf(TabiTokenizer)
        print_result("TabiTokenizer", ids, decoded, pieces)
    except Exception as exc:
        print(f"  SKIPPED — could not load model: {exc}")

    print(f"\n{SEPARATOR}")
    print("  Done.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
