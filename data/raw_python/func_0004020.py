def cns_vwl_pstr_long(self):
        """
        Return a new IPAString, containing only:
        
        1. the consonants,
        2. the vowels, and
        3. the primary stress diacritics, and
        4. the long suprasegmental

        in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if (c.is_letter) or (c.is_suprasegmental and (c.is_primary_stress or c.is_long))])