def _do_word(self, word, lemma=None, type=None):
        """ Adds a new Word to the sentence.
            Other Sentence._do_[tag] functions assume a new word has just been appended.
        """
        # Improve 3rd person singular "'s" lemma to "be", e.g., as in "he's fine".
        if lemma == "'s" and type in ("VB", "VBZ"):
            lemma = "be"
        self.words.append(Word(self, word, lemma, type, index=len(self.words)))