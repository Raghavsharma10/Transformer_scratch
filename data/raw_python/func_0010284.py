def get_json(self, path, **kwargs):
        """Perform an HTTP GET request with JSON headers of the specified path against Device Cloud

        Make an HTTP GET request against Device Cloud with this accounts
        credentials and base url.  This method uses the
        `requests <http://docs.python-requests.org/en/latest/>`_ library
        `request method <http://docs.python-requests.org/en/latest/api/#requests.request>`_
        and all keyword arguments will be passed on to that method.

        This method will automatically add the ``Accept: application/json`` and parse the
        JSON response from Device Cloud.

        :param str path: Device Cloud path to GET
        :param int retries: The number of times the request should be retried if an
            unsuccessful response is received.  Most likely, you should leave this at 0.
        :raises DeviceCloudHttpException: if a non-success response to the request is received
            from Device Cloud
        :returns: A python data structure containing the results of calling ``json.loads`` on the
            body of the response from Device Cloud.

        """

        url = self._make_url(path)
        headers = kwargs.setdefault('headers', {})
        headers.update({'Accept': 'application/json'})
        response = self._make_request("GET", url, **kwargs)
        return json.loads(response.text)