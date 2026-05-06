def highPass(self, *args):
        """
        Creates a copy of the signal with the high pass applied, args specifed are passed through to _butter. 
        :return: 
        """
        return Signal(self._butter(self.samples, 'high', *args), fs=self.fs)