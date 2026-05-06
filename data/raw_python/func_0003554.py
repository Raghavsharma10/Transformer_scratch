def _init(self):
        """Read the line number."""
        self._line_number = next_line(
            self._communication.last_requested_line_number,
            self._file.read(1)[0])