def move_cursor_one_word(self, word=LEFT):
        """Move the cursor of one word to the right (1) or the the left (-1)."""

        assert word in (self.RIGHT, self.LEFT)

        if word == self.RIGHT:
            papy = self.text.find(' ', self.cursor) + 1
            if not papy:
                papy = len(self)
            self.cursor = papy
        else:
            papy = self.text.rfind(' ', 0, self.cursor)
            if papy == -1:
                papy = 0
            self.cursor = papy