"""
TurkishTokenizer implementation following Algorithm 1 (encode) and
Algorithm 2 (decode) of

    Bayram, M. A., Fincan, A. A., Gümüş, A. S., Karakaş, S., Diri, B.,
    Yıldırım, S., Çelik, D. "Tokens with Meaning: A Hybrid Tokenization
    Approach for Turkish", 2025.
"""

import string

from .BaseTokenizer import BaseTokenizer


class TurkishTokenizer(BaseTokenizer):
    ID_UPPERCASE = "<uppercase>"
    ID_UNK = "<unk>"
    ID_SPACE = "<space>"
    _PUNCT_PREFIX = "<P:"
    _PUNCT_SUFFIX = ">"

    def __init__(self, morph_analyzer=None, bpe_fallback=None):
        self.morph_analyzer = morph_analyzer
        self.bpe_fallback = bpe_fallback

    # ENCODE
    def _preprocess(self, raw_text):
        text = raw_text
        for p in string.punctuation:
            text = text.replace(p, " " + p + " ")
        return text

    def encode(self, text):
        if self.morph_analyzer is None:
            raise ValueError("TurkishTokenizer requires a morph_analyzer.")

        token_ids = []
        processed = self._preprocess(text)

        for w in processed.split():
            if len(w) == 1 and w in string.punctuation:
                token_ids.append(self._punct_id(w))
                continue

            if self._is_capitalized(w):
                token_ids.append(self.ID_UPPERCASE)
                w = self._turkish_lower(w)

            root_id, suffix_ids = self.morph_analyzer.analyze(w)
            if root_id is not None:
                token_ids.append(root_id)
                token_ids.extend(suffix_ids)
            elif self.bpe_fallback is not None:
                token_ids.extend(self.bpe_fallback.encode(w))
            else:
                token_ids.append(self.ID_UNK)

        return token_ids

    # DECODE

    def decode(self, token_ids):
        parts = []
        i = 0
        n = len(token_ids)

        while i < n:
            tid = token_ids[i]

            if tid == self.ID_UPPERCASE and i + 1 < n:
                next_tid = token_ids[i + 1]
                base = self._surface_of(next_tid, parts)
                lead = ""
                while base.startswith(" "):
                    lead += " "
                    base = base[1:]
                parts.append(lead + self._turkish_capitalize(base))
                i += 2
                continue

            candidates = self.morph_analyzer.reverse_lookup(tid)
            if len(candidates) > 1:
                ctx = self.morph_analyzer.get_vowel_context(parts)
                surface = self.morph_analyzer.apply_phonology(tid, ctx, parts)
            else:
                surface = candidates[0]

            parts.append(self._unwrap_special(surface))
            i += 1

        return "".join(parts)

    # helpers

    def _surface_of(self, tid, parts):
        candidates = self.morph_analyzer.reverse_lookup(tid)
        if len(candidates) > 1:
            ctx = self.morph_analyzer.get_vowel_context(parts)
            return self.morph_analyzer.apply_phonology(tid, ctx, parts)
        return self._unwrap_special(candidates[0])

    @classmethod
    def _punct_id(cls, char):
        return cls._PUNCT_PREFIX + char + cls._PUNCT_SUFFIX

    @classmethod
    def _unwrap_special(cls, surface):
        if isinstance(surface, str):
            if surface.startswith(cls._PUNCT_PREFIX) and surface.endswith(cls._PUNCT_SUFFIX):
                return surface[len(cls._PUNCT_PREFIX):-len(cls._PUNCT_SUFFIX)]
            if surface == cls.ID_SPACE:
                return " "
        return surface

    # Turkish-aware casing

    _TR_LOWER = str.maketrans({"İ": "i", "I": "ı"})
    _TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})

    @classmethod
    def _turkish_lower(cls, word):
        return word.translate(cls._TR_LOWER).lower()

    @classmethod
    def _turkish_upper_char(cls, ch):
        return ch.translate(cls._TR_UPPER).upper()

    @classmethod
    def _is_capitalized(cls, word):
        if not word:
            return False
        first = word[0]
        return first.isupper() or first == "İ"

    @classmethod
    def _turkish_capitalize(cls, word):
        if not word:
            return word
        return cls._turkish_upper_char(word[0]) + word[1:]
