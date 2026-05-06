def set_starting_ratio(self, ratio):
        """ Set the starting conversion ratio for the next `read` call. """
        from samplerate.lowlevel import src_set_ratio
        if self._state is None:
            self._create()
        src_set_ratio(self._state, ratio)
        self.ratio = ratio