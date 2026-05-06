def cursor_query(self, collection, query):
        """
        :param str collection: The name of the collection for the request.
        :param dict query: Dictionary of solr args.

        Will page through the result set in increments using cursorMark until it has all items. Sort is required for cursorMark \
        queries, if you don't specify it, the default is 'id desc'.

        Returns an iterator of SolrResponse objects. For Example::

            >>> for res in solr.cursor_query('SolrClient_unittest',{'q':'*:*'}):
                    print(res)
        """
        cursor = '*'
        if 'sort' not in query:
            query['sort'] = 'id desc'
        while True:
            query['cursorMark'] = cursor
            # Get data with starting cursorMark
            results = self.query(collection, query)
            if results.get_results_count():
                cursor = results.get_cursor()
                yield results
            else:
                self.logger.debug("Got zero Results with cursor: {}".format(cursor))
                break