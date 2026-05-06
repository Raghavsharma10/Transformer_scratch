def delete_one_word(self, word=RIGHT):
        """Delete one word the right or the the left of the cursor."""

        assert word in (self.RIGHT, self.LEFT)

        if word == self.RIGHT:
            papy = self.text.find(' ', self.cursor) + 1
            if not papy:
                papy = len(self.text)
            self.text = self.text[:self.cursor] + self.text[papy:]

        else:
            papy = self.text.rfind(' ', 0, self.cursor)
            if papy == -1:
                papy = 0
            self.text = self.text[:papy] + self.text[self.cursor:]
            self.cursor = papy