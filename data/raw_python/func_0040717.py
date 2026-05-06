def set_word_indices(self, wordlist):
        """
        Populate the list of word_indices, mapping self.words to the given wordlist
        """
        self.word_indices = [wordlist.index(word) for word in self.words]