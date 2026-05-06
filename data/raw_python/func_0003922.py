def dump(self, title, coordinates):
        """Dump a frame to the trajectory file

           Arguments:
            | ``title``  --  the title of the frame
            | ``coordinates``  --  a numpy array with coordinates in atomic units
        """
        print("% 8i" % len(self.symbols), file=self._f)
        print(str(title), file=self._f)
        for symbol, coordinate in zip(self.symbols, coordinates):
            print("% 2s % 12.9f % 12.9f % 12.9f" % ((symbol, ) + tuple(coordinate/self.file_unit)), file=self._f)