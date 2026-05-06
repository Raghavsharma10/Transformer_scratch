def vowels(self):
        """
        Return a new IPAString, containing only the vowels in the current string.

        :rtype: IPAString
        """
        return IPAString(ipa_chars=[c for c in self.ipa_chars if c.is_vowel])