def query(self):
        """Return all start records for this the dataset, grouped by the start record"""

        return self._session.query(Process).filter(Process.d_vid == self._d_vid)