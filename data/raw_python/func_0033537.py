def get_metric_names(self, agent_id, re=None, limit=5000):
        """
        Requires: application ID
        Optional: Regex to filter metric names, limit of results
        Returns: A dictionary,
                    key:  metric name,
                    value: list of fields available for a given metric
        Method: Get
        Restrictions: Rate limit to 1x per minute
        Errors: 403 Invalid API Key, 422 Invalid Parameters
        Endpoint: api.newrelic.com
        """
        # Make sure we play it slow
        self._api_rate_limit_exceeded(self.get_metric_names)

        # Construct our GET request parameters into a nice dictionary
        parameters = {'re': re, 'limit': limit}

        endpoint = "https://api.newrelic.com"
        uri = "{endpoint}/api/v1/applications/{agent_id}/metrics.xml"\
              .format(endpoint=endpoint, agent_id=agent_id)

        # A longer timeout is needed due to the amount of
        # data that can be returned without a regex search
        response = self._make_get_request(uri, parameters=parameters, timeout=max(self.timeout, 5.0))

        # Parse the response. It seems clearer to return a dict of
        # metrics/fields instead of a list of metric objects. It might be more
        # consistent with the retrieval of metric data to make them objects but
        # since the attributes in each type of metric object are different
        # (and we aren't going to make heavyweight objects) we don't want to.
        metrics = {}
        for metric in response.findall('.//metric'):
            fields = []
            for field in metric.findall('.//field'):
                fields.append(field.get('name'))
            metrics[metric.get('name')] = fields
        return metrics