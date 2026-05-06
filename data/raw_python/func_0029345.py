def pop_aggregations_params(self):
        """ Pop and return aggregation params from query string params.

        Aggregation params are expected to be prefixed(nested under) by
        any of `self._aggregations_keys`.
        """
        from nefertari.view import BaseView
        self._query_params = BaseView.convert_dotted(self.view._query_params)

        for key in self._aggregations_keys:
            if key in self._query_params:
                return self._query_params.pop(key)
        else:
            raise KeyError('Missing aggregation params')