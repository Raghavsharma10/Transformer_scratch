def _setup_aggregation(self, aggregator=None):
        """ Wrap `self.index` method with ESAggregator.

        This makes `self.index` to first try to run aggregation and only
        on fail original method is run. Method is wrapped only if it is
        defined and `elasticsearch.enable_aggregations` setting is true.
        """
        from nefertari.elasticsearch import ES
        if aggregator is None:
            aggregator = ESAggregator
        aggregations_enabled = (
            ES.settings and ES.settings.asbool('enable_aggregations'))
        if not aggregations_enabled:
            log.debug('Elasticsearch aggregations are not enabled')
            return

        index = getattr(self, 'index', None)
        index_defined = index and index != self.not_allowed_action
        if index_defined:
            self.index = aggregator(self).wrap(self.index)