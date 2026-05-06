def loadSignal(self, name, start=None, end=None):
        """
        Loads the named entry from the upload cache as a signal.
        :param name: the name.
        :param start: the time to start from in HH:mm:ss.SSS format
        :param end: the time to end at in HH:mm:ss.SSS format.
        :return: the signal if the named upload exists.
        """
        entry = self._getCacheEntry(name)
        if entry is not None:
            from analyser.common.signal import loadSignalFromWav
            return loadSignalFromWav(entry['path'], start=start, end=end)
        else:
            return None