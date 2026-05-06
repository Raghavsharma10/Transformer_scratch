def sounds_like(self, word1, word2):
        """Compare the phonetic representations of 2 words, and return a boolean value."""
        return self.phonetics(word1) == self.phonetics(word2)