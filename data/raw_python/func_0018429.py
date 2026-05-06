def move_cursor_one_letter(self, letter=RIGHT):
        """Move the cursor of one letter to the right (1) or the the left."""
        assert letter in (self.RIGHT, self.LEFT)

        if letter == self.RIGHT:
            self.cursor += 1
            if self.cursor > len(self.text):
                self.cursor -= 1
        else:
            self.cursor -= 1
            if self.cursor < 0:
                self.cursor += 1