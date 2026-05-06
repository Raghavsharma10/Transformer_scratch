def send(self, request):
        """ Send a request to the server and wait for its response.

        Args:
            request (Request): Reference to a request object that is sent to the server.

        Returns:
            Response: The response from the server to the request.
        """
        self._connection.connection.rpush(self._request_key, pickle.dumps(request))
        resp_key = '{}:{}'.format(SIGNAL_REDIS_PREFIX, request.uid)

        while True:
            if self._connection.polling_time > 0.0:
                sleep(self._connection.polling_time)

            response_data = self._connection.connection.get(resp_key)
            if response_data is not None:
                self._connection.connection.delete(resp_key)
                break

        return pickle.loads(response_data)