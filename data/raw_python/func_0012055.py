def scroll(self, query=None, scroll='5m', size=100, unpack=True):
        """Scroll an index with the specified search query.

        Works as a generator. Will yield `size` results per iteration until all hits are returned.
        """
        query = self.match_all if query is None else query
        response = self.instance.search(index=self.index, doc_type=self.doc_type, body=query, size=size, scroll=scroll)
        while len(response['hits']['hits']) > 0:
            scroll_id = response['_scroll_id']
            logging.debug(response)
            if unpack:
                yield [source['_source'] if '_source' in source else source for source in response['hits']['hits']]
            else:
                yield response['hits']['hits']
            response = self.instance.scroll(scroll_id=scroll_id, scroll=scroll)