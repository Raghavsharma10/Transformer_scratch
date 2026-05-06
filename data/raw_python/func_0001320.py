def clear_last_lines(self, n):
        """Clear last N lines of terminal output.
        """
        self.term.stream.write(
            self.term.move_up * n + self.term.clear_eos)
        self.term.stream.flush()