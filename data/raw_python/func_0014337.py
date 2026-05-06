def all(self, usage_type, usage_period_id, api, query=None, *args, **kwargs):
        """
        Gets all api usages by type for a given period an api.
        """

        if query is None:
            query = {}

        mandatory_query = {
            'filters[usagePeriod]': usage_period_id,
            'filters[metric]': api
        }

        mandatory_query.update(query)

        return self.client._get(
            self._url(usage_type),
            mandatory_query,
            headers={
                'x-contentful-enable-alpha-feature': 'usage-insights'
            }
        )