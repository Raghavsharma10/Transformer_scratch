def _read_frame(self):
        """Read a frame from the XYZ file"""

        size = self.read_size()
        title = self._f.readline()[:-1]
        if self.symbols is None:
            symbols = []
        coordinates = np.zeros((size, 3), float)
        for counter in range(size):
            line = self._f.readline()
            if len(line) == 0:
                raise StopIteration
            words = line.split()
            if len(words) < 4:
                raise StopIteration
            if self.symbols is None:
                symbols.append(words[0])
            try:
                coordinates[counter, 0] = float(words[1])
                coordinates[counter, 1] = float(words[2])
                coordinates[counter, 2] = float(words[3])
            except ValueError:
                raise StopIteration
        coordinates *= self.file_unit
        if self.symbols is None:
            self.symbols = symbols
        return title, coordinates