def _read_section(self):
        """Read and return an entire section"""
        lines = [self._last[self._last.find(":")+1:]]
        self._last = self._f.readline()
        while len(self._last) > 0 and len(self._last[0].strip()) == 0:
            lines.append(self._last)
            self._last = self._f.readline()
        return lines