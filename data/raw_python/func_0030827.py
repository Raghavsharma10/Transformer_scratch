def _index_document(self, document, force=False):
        """ Adds parition document to the index. """
        from ambry.util import int_maybe
        time_coverage = document.pop('time_coverage', [])
        from_year = None
        to_year = None
        if time_coverage:
            from_year = int_maybe(time_coverage[0])
            to_year = int_maybe(time_coverage[-1])

        query = text("""
            INSERT INTO partition_index(vid, dataset_vid, title, keywords, doc, from_year, to_year)
            VALUES(:vid, :dataset_vid, :title, :keywords, :doc, :from_year, :to_year); """)
        self.backend.library.database.connection.execute(
            query, from_year=from_year, to_year=to_year, **document)