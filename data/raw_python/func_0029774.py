def _index_document(self, identifier, force=False):
        """ Adds identifier document to the index. """

        query = text("""
            INSERT INTO identifier_index(identifier, type, name)
            VALUES(:identifier, :type, :name);
        """)
        self.execute(query, **identifier)