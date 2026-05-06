def update_file(self):
        """Update the read-in configuration file.
        """
        if self._filename is None:
            raise NoConfigFileReadError()
        with open(self._filename, 'w') as fb:
            self.write(fb)