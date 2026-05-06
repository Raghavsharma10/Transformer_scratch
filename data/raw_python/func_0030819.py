def list_documents(self, limit=None):
        """ Generates vids of all indexed identifiers.

        Args:
            limit (int, optional): If not empty, the maximum number of results to return

        Generates:
            str: vid of the document.
        """
        limit_str = ''
        if limit:
            try:
                limit_str = 'LIMIT {}'.format(int(limit))
            except (TypeError, ValueError):
                pass

        query = ('SELECT identifier FROM identifier_index ' + limit_str)

        for row in self.backend.library.database.connection.execute(query).fetchall():
            yield row['identifier']