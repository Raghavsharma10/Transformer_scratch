def save(self, filename, callback=_noop_callback):
        """Save this file.

        :param filename: the file to which to save the .sav file
        :type filename: str
        :param callback: a progress callback function
        :type callback: function
        """
        with open(filename, 'wb') as fp:
            self._save(fp, callback)