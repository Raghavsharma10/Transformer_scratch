def _index_document(self, document, force=False):
        """ Adds dataset document to the index. """
        query = text("""
            INSERT INTO dataset_index(vid, title, keywords, doc)
            VALUES(:vid, :title, string_to_array(:keywords, ' '), to_tsvector('english', :doc));
        """)
        self.execute(query, **document)