def _index_document(self, document, force=False):
        """ Adds document to the index. """
        query = text("""
            INSERT INTO dataset_index(vid, title, keywords, doc)
            VALUES(:vid, :title, :keywords, :doc);
        """)
        self.backend.library.database.connection.execute(query, **document)