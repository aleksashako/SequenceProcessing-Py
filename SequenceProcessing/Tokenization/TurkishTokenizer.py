from .BaseTokenizer import BaseTokenizer
import string

class TurkishTokenizer(BaseTokenizer):
    def __init__(self, morph_analyzer=None, bpe_fallback=None):
        """
        Initialization. 
        """
        self.morph_analyzer = morph_analyzer
        self.bpe_fallback = bpe_fallback
        
        # specific tokens according to the article
        self.ID_UPPERCASE = "<uppercase>"

    def _preprocess(self, raw_text: str) -> str:
        """
        Separates punctuation marks with spaces so that the loop sees them as separate tokens
        """
        text = raw_text
        for p in string.punctuation:
            text = text.replace(p, f" {p} ")
        return text

    def encode(self, text: str) -> list[int]:
        """
        TurkishTokenizer Tokenization Pipeline
        """
        token_ids = []
        
        processed_text = self._preprocess(text)
        
        for w in processed_text.split():
            
            if w in string.punctuation:
                token_ids.append(w) 
                continue
                
            if w[0].isupper():
                token_ids.append(self.ID_UPPERCASE)
                w = w.lower() 
                
            # returns root ID anf a list of suffix ID 
            root_id, suffix_ids = self.morph_analyzer.analyze(w)
            
            if root_id is not None:
                token_ids.append(root_id)
                token_ids.extend(suffix_ids)
            else:
                subwords = self.bpe_fallback.encode(w)
                token_ids.extend(subwords)
                
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        TurkishTokenizer Decoding Pipeline
        Reconstructs the original text string from a sequence of token IDs.
        """
        parts = []
        i = 0
        total_tokens = len(token_ids)

        while i < total_tokens:
            token_id = token_ids[i]

            if token_id == self.ID_UPPERCASE:
                next_token_id = token_ids[i + 1]
                
                base_word = self.morph_analyzer.lookup_base_string(next_token_id)
                
                # Capitalize it using Turkish rules and append
                capitalized_word = self._turkish_capitalize(base_word)
                parts.append(capitalized_word)
                
                i += 2
                continue

            # Get surface form variants (candidates)
            candidates = self.morph_analyzer.reverse_lookup(token_id)

            if len(candidates) > 1:
                ctx = self.morph_analyzer.get_vowel_context(parts)
                
                # Apply phonology rules to select the correct variant
                surface = self.morph_analyzer.apply_phonology(token_id, ctx, token_ids, i)
            
            else:
                surface = candidates[0]

            parts.append(surface)
            i += 1

        return "".join(parts)

    def _turkish_capitalize(self, word: str) -> str:
        """
        Helper function to handle Turkish-specific capitalization.
        Ensures 'i' becomes 'İ' and 'ı' becomes 'I'.
        """
        if not word:
            return word
            
        first_char = word[0]
        if first_char == 'i':
            return 'İ' + word[1:]
        elif first_char == 'ı':
            return 'I' + word[1:]
        else:
            return first_char.upper() + word[1:]