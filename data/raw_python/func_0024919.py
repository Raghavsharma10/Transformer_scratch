def _get_datapoints(self, params):
        """
        Will make a direct REST call with the given json body payload to
        get datapoints.
        """
        url = self.query_uri + '/v1/datapoints'
        return self.service._get(url, params=params)