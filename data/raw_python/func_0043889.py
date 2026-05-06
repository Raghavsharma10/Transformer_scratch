def serialize_search(self, pid_fetcher, search_result,
                         item_links_factory=None, **kwargs):
        """Serialize a search result.

        :param pid_fetcher: Persistent identifier fetcher.
        :param search_result: Elasticsearch search result.
        :param item_links_factory: Factory function for the items in result.
            (Default: ``None``)
        :returns: The objects serialized.
        """
        ret = [self.transform_search_hit(pid_fetcher(hit['_id'],
                                         hit['_source']),
                                         hit,
                                         links_factory=item_links_factory)
               for hit in search_result['hits']['hits']]

        return dumps(ret, **self.dumps_kwargs)