def append_line(self, new_line):
        """Appends the new_line to the LS340 program."""
        # TODO: The user still has to write the raw line, this is error prone.
        self._write(('PGM', [Integer, String]), self.idx, new_line)