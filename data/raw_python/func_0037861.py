def add(self, documents, boost=None):
        """
        Adds documents to Solr index
        documents - Single item or list of items to add
        """

        if not isinstance(documents, list):
            documents = [documents]
        documents = [{'doc': d} for d in documents]
        if boost:
            for d in documents:
                d['boost'] = boost

        self._add_batch.extend(documents)

        if len(self._add_batch) > SOLR_ADD_BATCH:
            self._addFlushBatch()