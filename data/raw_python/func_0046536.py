def to_query(self):
        """
        Iterates over all filters and converts them to an Elastic HTTP API
        suitable query.

        Note: each :class:`~es_fluent.filters.Filter` is free to set it's own
        filter dictionary. ESFluent does not attempt to guard against filters
        that may clobber one another.  If you wish to ensure that filters are
        isolated, nest them inside of a boolean filter such as
        :class:`~es_fluent.filters.core.And` or
        :class:`~es_fluent.filters.core.Or`.
       """
        query = {}
        for filter_instance in self.filters:
            if filter_instance.is_empty():
                continue
            filter_query = filter_instance.to_query()
            query.update(filter_query)

        return query