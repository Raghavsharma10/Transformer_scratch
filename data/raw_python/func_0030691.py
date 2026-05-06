def clean(self):
        """Delete all of the records"""

        # Deleting seems to be really weird and unrelable.
        self._session \
            .query(Process) \
            .filter(Process.d_vid == self._d_vid) \
            .delete(synchronize_session='fetch')

        for r in self.records:
            self._session.delete(r)

        self._session.commit()