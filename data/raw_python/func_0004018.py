def cns_vwl_str(self):
        """
        Return a new IPAString, containing only:
        
        1. the consonants,
        2. the vowels, and
        3. the stress diacritics

        in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if (c.is_letter) or (c.is_suprasegmental and c.is_stress)])