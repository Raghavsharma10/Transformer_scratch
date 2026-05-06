def _skip_frame(self):
        """Skip a single frame from the trajectory"""
        size = self.read_size()
        for i in range(size+1):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration