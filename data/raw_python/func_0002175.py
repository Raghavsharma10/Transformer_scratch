def _read_header(self):
        """Get the needed header information to initialize dataset."""
        self._header = self.cdmrf.fetch_header()
        self.load_from_stream(self._header)