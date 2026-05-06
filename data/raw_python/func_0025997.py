def lowPass(self, *args):
        """
        Creates a copy of the signal with the low pass applied, args specifed are passed through to _butter. 
        :return: 
        """
        return Signal(self._butter(self.samples, 'low', *args), fs=self.fs)