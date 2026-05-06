def _skip_section(self):
        """Skip a section"""
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            self._last = self._f.readline()