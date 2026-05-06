def paging_query(self, collection, query, rows=1000, start=0, max_start=200000):
        """
        :param str collection: The name of the collection for the request.
        :param dict query: Dictionary of solr args.
        :param int rows: Number of rows to return in each batch. Default is 1000.
        :param int start: What position to start with. Default is 0.
        :param int max_start: Once the start will reach this number, the function will stop. Default is 200000.

        Will page through the result set in increments of `row` WITHOUT using cursorMark until it has all items \
        or until `max_start` is reached. Use max_start to protect your Solr instance if you are not sure how many items you \
        will be getting. The default is 200,000, which is still a bit high.

        Returns an iterator of SolrResponse objects. For Example::

            >>> for res in solr.paging_query('SolrClient_unittest',{'q':'*:*'}):
                    print(res)

        """
        query = dict(query)
        while True:
            query['start'] = start
            query['rows'] = rows
            res = self.query(collection, query)
            if res.get_results_count():
                yield res
                start += rows
            if res.get_results_count() < rows or start > max_start:
                break