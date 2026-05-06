def _delete(self, vid=None):
        """ Deletes given dataset from index.

        Args:
            vid (str): dataset vid.

        """

        assert vid is not None, 'vid argument can not be None.'
        writer = self.index.writer()
        writer.delete_by_term('vid', vid)
        writer.commit()