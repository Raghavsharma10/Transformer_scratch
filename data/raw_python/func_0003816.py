def _get_line(self):
        """Get a line or raise StopIteration"""
        line = self._f.readline()
        if len(line) == 0:
            raise StopIteration
        return line