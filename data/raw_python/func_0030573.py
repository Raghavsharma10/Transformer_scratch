def _delete(self, identifier=None):
        """ Deletes given identifier from index.

        Args:
            identifier (str): identifier of the document to delete.

        """
        assert identifier is not None, 'identifier argument can not be None.'
        writer = self.index.writer()
        writer.delete_by_term('identifier', identifier)
        writer.commit()