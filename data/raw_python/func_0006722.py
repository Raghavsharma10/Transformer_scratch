def add_word(self, word):
        """
        Add a word to the GADDAG.

        Args:
            word: A word to be added to the GADDAG.
        """
        word = word.lower()

        if not (word.isascii() and word.isalpha()):
            raise ValueError("Invalid character in word '{}'".format(word))

        word = word.encode(encoding="ascii")
        result = cgaddag.gdg_add_word(self.gdg, word)
        if result == 1:
            raise ValueError("Invalid character in word '{}'".format(word))
        elif result == 2:
            raise MemoryError("Out of memory, GADDAG is in an undefined state")