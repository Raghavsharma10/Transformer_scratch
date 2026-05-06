def _index_document(self, document, force=False):
        """ Adds parition document to the index. """

        time_coverage = document.pop('time_coverage', [])
        from_year = None
        to_year = None
        if time_coverage:
            from_year = int(time_coverage[0]) if time_coverage and time_coverage[0] else None
            to_year = int(time_coverage[-1]) if time_coverage and time_coverage[-1] else None

        query = text("""
            INSERT INTO partition_index(vid, dataset_vid, title, keywords, doc, from_year, to_year)
            VALUES(
                :vid, :dataset_vid, :title,
                string_to_array(:keywords, ' '),
                to_tsvector('english', :doc),
                :from_year, :to_year); """)

        self.execute(query, from_year=from_year, to_year=to_year, **document)