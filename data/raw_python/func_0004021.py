def cns_vwl_str_len_wb(self):
        """
        Return a new IPAString, containing only:
        
        1. the consonants,
        2. the vowels, and
        3. the stress diacritics,
        4. the length diacritics, and
        5. the word breaks

        in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if (c.is_letter) or (c.is_suprasegmental and (c.is_stress or c.is_length or c.is_word_break))])