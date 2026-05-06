def cns_vwl(self):
        """
        Return a new IPAString, containing only:
        
        1. the consonants, and
        2. the vowels

        in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if c.is_letter])