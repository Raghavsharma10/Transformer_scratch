def get_threshold_values(self, application_id):
        """
        Requires: account ID, list of application ID
        Method: Get
        Endpoint: api.newrelic.com
        Restrictions: ???
        Errors: 403 Invalid API key, 422 Invalid Parameters
        Returns: A list of threshold_value objects, each will have information
                 about its start/end time, metric name, metric value, and the
                 current threshold
        """
        endpoint = "https://rpm.newrelic.com"
        remote_file = "threshold_values.xml"
        uri = "{endpoint}/accounts/{account_id}/applications/{app_id}/{xml}".format(endpoint=endpoint, account_id=self.account_id, app_id=application_id, xml=remote_file)
        response = self._make_get_request(uri)
        thresholds = []

        for threshold_value in response.findall('.//threshold_value'):
            properties = {}
            # a little ugly, but the output works fine.
            for tag, text in threshold_value.items():
                properties[tag] = text
            thresholds.append(Threshold(properties))
        return thresholds