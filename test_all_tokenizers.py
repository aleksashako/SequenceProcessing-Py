"""
Full comparison test: runs all four tokenizers across multiple Turkish
sentences and prints per-sentence results plus a summary table.

TurkishTokenizer runs fully offline (dummy morphological resources).
CosmosGPT2, Mursit, and TabiBERT require a network connection on first
use to download model files; each is skipped gracefully if unavailable.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from SequenceProcessing.Tokenization.MorphologicalAnalyzer import MorphologicalAnalyzer
from SequenceProcessing.Tokenization.TurkishTokenizer import TurkishTokenizer
from SequenceProcessing.Tokenization.CosmosGPT2Tokenizer import CosmosGPT2Tokenizer
from SequenceProcessing.Tokenization.MursitTokenizer import MursitTokenizer
from SequenceProcessing.Tokenization.TabiTokenizer import TabiTokenizer

# ── Test sentences ────────────────────────────────────────────────────────────
# Each covers a different aspect of Turkish morphology / tokenization challenge.
TEST_CASES = [
    (
        "arabalar garajda duruyor.",
        "root + plural suffix + locative suffix + progressive verb",
    ),
    (
        "Türkiye Cumhuriyeti'nin başkenti Ankara'dır.",
        "proper nouns, genitive, clitic apostrophe",
    ),
    (
        "Yapay zeka teknolojileri hızla gelişiyor.",
        "OOV / technical vocabulary, adverb, progressive",
    ),
    (
        "İstanbul dünyanın en güzel şehirlerinden biridir.",
        "superlative, ablative plural, copula",
    ),
    (
        "Kitabı okuyorum, ama hiçbir şey anlamıyorum!",
        "accusative, first-person progressive, negation, punctuation",
    ),
]

SEP  = "=" * 72
THIN = "─" * 72


# ── Helpers ───────────────────────────────────────────────────────────────────

class DummyBPE:
    """BPE fallback for TurkishTokenizer: echoes the unknown word as one token."""
    def encode(self, text: str):
        return [text]


def make_turkish():
    morph = MorphologicalAnalyzer(roots_path="dummy", affixes_path="dummy")
    return TurkishTokenizer(morph_analyzer=morph, bpe_fallback=DummyBPE())


def tokenize_all(sentence: str, tok_turkish, cosmos, mursit, tabi):
    """Return (name, ids, pieces_or_None, decoded, ok) for each tokenizer."""
    results = []

    # 1. TurkishTokenizer
    try:
        ids     = tok_turkish.encode(sentence)
        decoded = tok_turkish.decode(ids)
        results.append(("TurkishTokenizer", ids, None, decoded))
    except Exception as exc:
        results.append(("TurkishTokenizer", [], None, f"ERROR: {exc}"))

    # 2. CosmosGPT2
    for name, tok in [
        ("CosmosGPT2Tokenizer", cosmos),
        ("MursitTokenizer",     mursit),
        ("TabiTokenizer",       tabi),
    ]:
        if tok is None:
            results.append((name, [], None, "SKIPPED"))
            continue
        try:
            ids     = tok.encode(sentence)
            pieces  = tok.convert_ids_to_tokens(ids)
            decoded = tok.decode(ids)
            results.append((name, ids, pieces, decoded))
        except Exception as exc:
            results.append((name, [], None, f"ERROR: {exc}"))

    return results


def round_trip_ok(original: str, decoded: str) -> bool:
    return original.replace(" ", "").lower() == decoded.replace(" ", "").lower()


def print_sentence_block(sentence: str, note: str, results: list):
    print(f"\n  Sentence : {sentence}")
    print(f"  Focus    : {note}")
    print(THIN)
    print(f"  {'Tokenizer':<24} {'#tok':>4}  {'Pieces / tokens':<35}  RT")
    print(THIN)
    for name, ids, pieces, decoded in results:
        n    = len(ids)
        ok   = "✓" if (ids and round_trip_ok(sentence, decoded)) else ("✗" if ids else "—")
        disp = str(pieces[:6])[1:-1] + ("…" if pieces and len(pieces) > 6 else "") if pieces else str(ids[:6])[1:-1]
        print(f"  {name:<24} {n:>4}  {disp:<35}  {ok}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  TURKISH TOKENIZER COMPARISON — DEMO TEST")
    print(SEP)

    # Load tokenizers once (lazy; HF models download on first encode call)
    tok_turkish = make_turkish()

    print("\nLoading HuggingFace tokenizers …")
    cosmos = mursit = tabi = None
    try:
        cosmos = CosmosGPT2Tokenizer()
        cosmos.encode("test")   # trigger download now, not mid-table
        print("  CosmosGPT2Tokenizer  ✓")
    except Exception as e:
        print(f"  CosmosGPT2Tokenizer  SKIPPED ({e})")

    try:
        mursit = MursitTokenizer()
        mursit.encode("test")
        print("  MursitTokenizer      ✓")
    except Exception as e:
        print(f"  MursitTokenizer      SKIPPED ({e})")

    try:
        tabi = TabiTokenizer()
        tabi.encode("test")
        print("  TabiTokenizer        ✓")
    except Exception as e:
        print(f"  TabiTokenizer        SKIPPED ({e})")

    # ── Per-sentence results ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-SENTENCE RESULTS")
    print(SEP)

    all_counts: dict[str, list[int]] = {
        "TurkishTokenizer": [],
        "CosmosGPT2Tokenizer": [],
        "MursitTokenizer": [],
        "TabiTokenizer": [],
    }

    for sentence, note in TEST_CASES:
        results = tokenize_all(sentence, tok_turkish, cosmos, mursit, tabi)
        print_sentence_block(sentence, note, results)
        for name, ids, _, _ in results:
            if ids:
                all_counts[name].append(len(ids))

    # ── Summary table ─────────────────────────────────────────────────────
    print(SEP)
    print("  SUMMARY  —  token counts per sentence")
    print(SEP)
    header = f"  {'Tokenizer':<24}" + "".join(f"  S{i+1:1d}" for i in range(len(TEST_CASES))) + "   avg"
    print(header)
    print(THIN)
    for name, counts in all_counts.items():
        if not counts:
            print(f"  {name:<24}  (not available)")
            continue
        cells = "".join(f"  {c:2d}" for c in counts)
        avg   = sum(counts) / len(counts)
        print(f"  {name:<24}{cells}  {avg:5.1f}")

    print()
    print("  S1–S5 map to the sentences above in order.")

    # ── MLM demo (Mursit only) ────────────────────────────────────────────
    if mursit is not None:
        print(f"\n{SEP}")
        print("  MURSIT MLM MASKING DEMO  (80/10/10 strategy, p=0.15)")
        print(SEP)
        demo_sent = TEST_CASES[1][0]   # sentence 2 — rich morphology
        batch = mursit.mlm_encode(demo_sent)
        masked  = mursit.convert_ids_to_tokens(batch["input_ids"])
        targets = mursit.convert_ids_to_tokens(
            batch["labels"][batch["labels"] != -100]
        )
        print(f"  Input   : {demo_sent}")
        print(f"  Masked  : {masked}")
        print(f"  Targets : {targets}  ({len(targets)} token(s) masked)")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
