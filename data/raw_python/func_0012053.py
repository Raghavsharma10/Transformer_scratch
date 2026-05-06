def search(self, query=None, size=100, unpack=True):
        """Search the index with a query.

        Can at most return 10'000 results from a search. If the search would yield more than 10'000 hits, only the first 10'000 are returned.
        The default number of hits returned is 100.
        """
        logging.info('Download all documents from index %s.', self.index)
        if query is None:
            query = self.match_all
        results = list()
        data = self.instance.search(index=self.index, doc_type=self.doc_type, body=query, size=size)
        if unpack:
            for items in data['hits']['hits']:
                if '_source' in items:
                    results.append(items['_source'])
                else:
                    results.append(items)
        else:
            results = data['hits']['hits']
        return results