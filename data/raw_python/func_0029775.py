def _delete(self, identifier=None):
        """ Deletes given identifier from index.

        Args:
            identifier (str): identifier of the document to delete.

        """
        query = text("""
            DELETE FROM identifier_index
            WHERE identifier = :identifier;
        """)
        self.execute(query, identifier=identifier)