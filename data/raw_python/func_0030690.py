def exceptions(self):
        """Return all start records for this the dataset, grouped by the start record"""

        return (self._session.query(Process)
                .filter(Process.d_vid == self._d_vid)
                .filter(Process.exception_class != None)
                .order_by(Process.modified)).all()