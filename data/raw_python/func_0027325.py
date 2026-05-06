def phase_type(self, value):
        '''compresses the waveform horizontally; one of
        ``"normal"``, ``"resync"``, ``"resync2"``'''
        self._params.phase_type = value
        self._overwrite_lock.disable()