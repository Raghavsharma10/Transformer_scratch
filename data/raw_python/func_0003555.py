def _init(self):
        """Read the success byte."""
        self._ready = self._file.read(1)
        self._hall_left = self._file.read(2)
        self._hall_right = self._file.read(2)
        self._carriage_type = self._file.read(1)[0]
        self._carriage_position = self._file.read(1)[0]