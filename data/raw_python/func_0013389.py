def _filterSearchFeaturesRequest(self, reference_name, gene_symbol, name,
                                     start, end):
        """
        formulate a sparql query string based on parameters
        """
        filters = []
        query = self._baseQuery()
        filters = []
        location = self._findLocation(reference_name, start, end)
        if location:
            filters.append("?feature = <{}>".format(location))
        if gene_symbol:
            filters.append('regex(?feature_label, "{}")')
        if name:
            filters.append(
                'regex(?feature_label, "{}")'.format(name))
        # apply filters
        filter = "FILTER ({})".format(' && '.join(filters))
        if len(filters) == 0:
            filter = ""
        query = query.replace("#%FILTER%", filter)
        return query