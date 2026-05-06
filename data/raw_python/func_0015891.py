def is_alive(self, vhost='%2F'):
        """
        Uses the aliveness-test API call to determine if the
        server is alive and the vhost is active. The broker (not this code)
        creates a queue and then sends/consumes a message from it.

        :param string vhost: There should be no real reason to ever change
            this from the default value, but it's there if you need to.
        :returns bool: True if alive, False otherwise
        :raises: HTTPError if *vhost* doesn't exist on the broker.

        """
        uri = Client.urls['live_test'] % vhost

        try:
            resp = self._call(uri, 'GET')
        except http.HTTPError as err:
            if err.status == 404:
                raise APIError("No vhost named '%s'" % vhost)
            raise

        if resp['status'] == 'ok':
            return True
        else:
            return False