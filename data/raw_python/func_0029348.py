def aggregate(self):
        """ Perform aggregation and return response. """
        from nefertari.elasticsearch import ES
        aggregations_params = self.pop_aggregations_params()
        if self.view._auth_enabled:
            self.check_aggregations_privacy(aggregations_params)
        self.stub_wrappers()

        return ES(self.view.Model.__name__).aggregate(
            _aggregations_params=aggregations_params,
            **self._query_params)