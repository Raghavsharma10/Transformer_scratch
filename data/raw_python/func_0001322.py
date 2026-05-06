def move_to(self, n):
        """Move back N lines in terminal.
        """
        self.term.stream.write(self.term.move_up * n)