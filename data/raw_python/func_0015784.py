def raw(self):
        """
        Build query and passes to Elasticsearch, then returns the raw
        format returned.
        """
        qs = self.build_search()
        es = self.get_es()

        index = self.get_indexes()
        doc_type = self.get_doctypes()

        if doc_type and not index:
            raise BadSearch(
                'You must specify an index if you are specifying doctypes.')

        extra_search_kwargs = {}
        if self.search_type:
            extra_search_kwargs['search_type'] = self.search_type

        hits = es.search(body=qs,
                         index=self.get_indexes(),
                         doc_type=self.get_doctypes(),
                         **extra_search_kwargs)

        log.debug('[%s] %s' % (hits['took'], qs))
        return hits