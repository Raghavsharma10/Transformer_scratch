def delete(self, path, retries=DEFAULT_THROTTLE_RETRIES, **kwargs):
        """Perform an HTTP DELETE request of the specified path in Device Cloud

        Make an HTTP DELETE request against Device Cloud with this accounts
        credentials and base url.  This method uses the
        `requests <http://docs.python-requests.org/en/latest/>`_ library
        `request method <http://docs.python-requests.org/en/latest/api/#requests.request>`_
        and all keyword arguments will be passed on to that method.

        :param str path: Device Cloud path to DELETE
        :param int retries: The number of times the request should be retried if an
            unsuccessful response is received.  Most likely, you should leave this at 0.
        :raises DeviceCloudHttpException: if a non-success response to the request is received
            from Device Cloud
        :returns: A requests ``Response`` object

        """
        url = self._make_url(path)
        return self._make_request("DELETE", url, **kwargs)