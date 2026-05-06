def delete_one_letter(self, letter=RIGHT):
        """Delete one letter the right or the the left of the cursor."""

        assert letter in (self.RIGHT, self.LEFT)

        if letter == self.LEFT:
            papy = self.cursor
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor = papy - 1

        else:
            self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]