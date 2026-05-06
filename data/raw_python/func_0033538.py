def get_metric_data(self, applications, metrics, field, begin, end, summary=False):
        """
        Requires: account ID,
                  list of application IDs,
                  list of metrics,
                  metric fields,
                  begin,
                  end
        Method: Get
        Endpoint: api.newrelic.com
        Restrictions: Rate limit to 1x per minute
        Errors: 403 Invalid API key, 422 Invalid Parameters
        Returns: A list of metric objects, each will have information about its
                 start/end time, application, metric name and any associated
                 values
        """
        # TODO: it may be nice to have some helper methods that make it easier
        #       to query by common time frames based off the time period folding
        #       of the metrics returned by the New Relic API.

        # Make sure we aren't going to hit an API timeout
        self._api_rate_limit_exceeded(self.get_metric_data)

        # Just in case the API needs parameters to be in order
        parameters = {}

        # Figure out what we were passed and set our parameter correctly
        # TODO: allow querying by something other than an application name/id,
        # such as server id or agent id
        try:
            int(applications[0])
        except ValueError:
            app_string = "app"
        else:
            app_string = "app_id"

        if len(applications) > 1:
            app_string = app_string + "[]"

        # Set our parameters
        parameters[app_string] = applications
        parameters['metrics[]'] = metrics
        parameters['field'] = field
        parameters['begin'] = begin
        parameters['end'] = end
        parameters['summary'] = int(summary)

        endpoint = "https://api.newrelic.com"
        uri = "{endpoint}/api/v1/accounts/{account_id}/metrics/data.xml"\
              .format(endpoint=endpoint, account_id=self.account_id)
        # A longer timeout is needed due to the
        # amount of data that can be returned
        response = self._make_get_request(uri, parameters=parameters, timeout=max(self.timeout, 5.0))

        # Parsing our response into lightweight objects and creating a list.
        # The dividing factor is the time period covered by the metric,
        # there should be no overlaps in time.
        metrics = []
        for metric in response.findall('.//metric'):
            metrics.append(Metric(metric))
        return metrics