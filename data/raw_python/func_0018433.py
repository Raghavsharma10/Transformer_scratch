def add_letter(self, letter):
        """Add a letter at the cursor pos."""
        assert isinstance(letter, str)
        assert len(letter) == 1

        self.text = self.text[:self.cursor] + letter + self.text[self.cursor:]
        self.cursor += 1