import json

class MorphologicalAnalyzer:
    def __init__(self, roots_path: str, affixes_path: str):
        """
        Loads the dictionaries provided by the authors of the paper.
        Expected files should contain:
        - 22,231 roots (mapping 'root' -> ID)
        - 177 suffix allomorphs mapped to 72 IDs.
        """
        # In a real scenario, uncomment these lines to load actual JSON files:
        # with open(roots_path, 'r', encoding='utf-8') as f:
        #     self.root_dict = json.load(f)  # e.g., {'araba': 'R_123', 'gel': 'R_456'}
        # with open(affixes_path, 'r', encoding='utf-8') as f:
        #     self.affix_dict = json.load(f) # e.g., {'lar': 'A_1', 'ler': 'A_1', 'mız': 'A_2'}

        # --- DUMMY DATA FOR TESTING ---
        # Example structure to test the logic before you get the actual files
        self.root_dict = {'araba': 'R_1', 'gel': 'R_2', 'kitap': 'R_3'}
        self.affix_dict = {'lar': 'A_1', 'ler': 'A_1', 'da': 'A_2', 'de': 'A_2'}
        # ------------------------------

        # Create reverse dictionaries for the Decoder (ID -> text)
        self.inv_root_dict = {v: k for k, v in self.root_dict.items()}
        
        # Suffix reverse dictionary must be a list since one ID has multiple forms
        self.inv_affix_dict = {}
        for surface_form, affix_id in self.affix_dict.items():
            if affix_id not in self.inv_affix_dict:
                self.inv_affix_dict[affix_id] = []
            self.inv_affix_dict[affix_id].append(surface_form)

        # Turkish vowels for phonology rules (Vowel Harmony)
        self.vowels = "aıoueiöü"
        self.back_vowels = "aıou"  
        self.front_vowels = "eiöü" 

 
    def analyze(self, word: str) -> tuple[str, list[str]]:
        """
        Greedy Dictionary Search (Line 13 in Algorithm 1).
        Finds the longest matching root, then splits the remainder into suffixes.
        Returns: (root_id, [suffix_id_1, suffix_id_2, ...]) or (None, [])
        """
        # Iterate backwards to find the longest matching root first
        for i in range(len(word), 0, -1):
            root_candidate = word[:i]
            
            if root_candidate in self.root_dict:
                root_id = self.root_dict[root_candidate]
                remainder = word[i:]
                
                if not remainder:
                    return root_id, []
                
                suffix_ids = self._extract_suffixes(remainder)
                if suffix_ids is not None:
                    return root_id, suffix_ids
                    
        return None, []

    def _extract_suffixes(self, remainder: str) -> list[str]:
        """
        Helper function: Greedy algorithm to split the remainder into known affix allomorphs.
        """
        suffixes = []
        current_str = remainder
        
        while current_str:
            found = False
            for length in range(len(current_str), 0, -1):
                suffix_candidate = current_str[:length]
                if suffix_candidate in self.affix_dict:
                    suffixes.append(self.affix_dict[suffix_candidate])
                    current_str = current_str[length:]
                    found = True
                    break
            
            if not found:
                return None
                
        return suffixes

    def lookup_base_string(self, token_id: str) -> str:
        """
        Returns the base text form of a root ID. Used for the uppercase rule.
        """
        return self.inv_root_dict.get(token_id, "")

    def reverse_lookup(self, token_id: str) -> list[str]:
        """
        Line 10: Get surface form variants (candidates).
        Returns a list of possible text representations for a given ID.
        """
        if token_id in self.inv_root_dict:
            return [self.inv_root_dict[token_id]]
        elif token_id in self.inv_affix_dict:
            return self.inv_affix_dict[token_id]
        else:
            return [str(token_id)] # Fallback for punctuation or unknown tokens

    def get_vowel_context(self, parts: list[str]) -> str:
        """
        Line 12: ctx <- GetVowelContext(parts).
        Finds the last vowel in the already decoded sequence for vowel harmony.
        """
        current_word_so_far = "".join(parts).lower()
        
        for char in reversed(current_word_so_far):
            if char in self.vowels:
                return char
        return "a" 

    def apply_phonology(self, token_id: str, last_vowel: str, all_tokens: list, current_idx: int) -> str:
        """
        Line 13: ApplyPhonology.
        Selects the correct allomorph based on Turkish Vowel Harmony rules.
        """
        candidates = self.inv_affix_dict[token_id]
        
        back_candidates = [c for c in candidates if self._get_first_vowel(c) in self.back_vowels]
        front_candidates = [c for c in candidates if self._get_first_vowel(c) in self.front_vowels]
        
        # Apply Major Vowel Harmony (Büyük Ünlü Uyumu)
        if last_vowel in self.back_vowels and back_candidates:
            return back_candidates[0]
        elif last_vowel in self.front_vowels and front_candidates:
            return front_candidates[0]
            
        return candidates[0]

    def _get_first_vowel(self, text: str) -> str:
        """Helper to find the first vowel in a suffix string."""
        for char in text.lower():
            if char in self.vowels:
                return char
        return "a"