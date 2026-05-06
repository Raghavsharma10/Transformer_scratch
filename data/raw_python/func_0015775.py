def build_search(self):
        """Builds the Elasticsearch search body represented by this S.

        Loop over self.steps to build the search body that will be
        sent to Elasticsearch. This returns a Python dict.

        If you want the JSON that actually gets sent, then pass the return
        value through :py:func:`elasticutils.utils.to_json`.

        :returns: a Python dict

        """
        filters = []
        filters_raw = None
        queries = []
        query_raw = None
        sort = []
        dict_fields = set()
        list_fields = set()
        facets = {}
        facets_raw = {}
        demote = None
        highlight_fields = set()
        highlight_options = {}
        suggestions = {}
        explain = False
        as_list = as_dict = False
        search_type = None

        for action, value in self.steps:
            if action == 'order_by':
                sort = []
                for key in value:
                    if isinstance(key, string_types) and key.startswith('-'):
                        sort.append({key[1:]: 'desc'})
                    else:
                        sort.append(key)
            elif action == 'values_list':
                if not value:
                    list_fields = set()
                else:
                    list_fields |= set(value)
                as_list, as_dict = True, False
            elif action == 'values_dict':
                if not value:
                    dict_fields = set()
                else:
                    dict_fields |= set(value)
                as_list, as_dict = False, True
            elif action == 'explain':
                explain = value
            elif action == 'query':
                queries.append(value)
            elif action == 'query_raw':
                query_raw = value
            elif action == 'demote':
                # value here is a tuple of (negative_boost, query)
                demote = value
            elif action == 'filter':
                filters.extend(self._process_filters(value))
            elif action == 'filter_raw':
                filters_raw = value
            elif action == 'facet':
                # value here is a (args, kwargs) tuple
                facets.update(_process_facets(*value))
            elif action == 'facet_raw':
                facets_raw.update(dict(value))
            elif action == 'highlight':
                if value[0] == (None,):
                    highlight_fields = set()
                else:
                    highlight_fields |= set(value[0])
                highlight_options.update(value[1])
            elif action == 'search_type':
                search_type = value
            elif action == 'suggest':
                suggestions[value[0]] = (value[1], value[2])
            elif action in ('es', 'indexes', 'doctypes', 'boost'):
                # Ignore these--we use these elsewhere, but want to
                # make sure lack of handling it here doesn't throw an
                # error.
                pass
            else:
                raise NotImplementedError(action)

        qs = {}

        # If there's a filters_raw, we use that.
        if filters_raw:
            qs['filter'] = filters_raw
        else:
            if len(filters) > 1:
                qs['filter'] = {'and': filters}
            elif filters:
                qs['filter'] = filters[0]

        # If there's a query_raw, we use that. Otherwise we use
        # whatever we got from query and demote.
        if query_raw:
            qs['query'] = query_raw

        else:
            pq = self._process_queries(queries)

            if demote is not None:
                qs['query'] = {
                    'boosting': {
                        'negative': self._process_queries([demote[1]]),
                        'negative_boost': demote[0]
                        }
                    }
                if pq:
                    qs['query']['boosting']['positive'] = pq

            elif pq:
                qs['query'] = pq

        if as_list:
            fields = qs['fields'] = list(list_fields) if list_fields else ['*']
        elif as_dict:
            fields = qs['fields'] = list(dict_fields) if dict_fields else ['*']
        else:
            fields = set()

        if facets:
            qs['facets'] = facets
            # Hunt for `facet_filter` shells and update those. We use
            # None as a shell, so if it's explicitly set to None, then
            # we update it.
            for facet in facets.values():
                if facet.get('facet_filter', 1) is None and 'filter' in qs:
                    facet['facet_filter'] = qs['filter']

        if facets_raw:
            qs.setdefault('facets', {}).update(facets_raw)

        if sort:
            qs['sort'] = sort
        if self.start:
            qs['from'] = self.start
        if self.stop is not None:
            qs['size'] = self.stop - self.start

        if highlight_fields:
            qs['highlight'] = self._build_highlight(
                highlight_fields, highlight_options)

        if explain:
            qs['explain'] = True

        for suggestion, (term, kwargs) in six.iteritems(suggestions):
            qs.setdefault('suggest', {})[suggestion] = {
                'text': term,
                'term': {
                    'field': kwargs.get('field', '_all'),
                },
            }

        self.fields, self.as_list, self.as_dict = fields, as_list, as_dict
        self.search_type = search_type
        return qs