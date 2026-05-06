def receive(self):
        """ Returns a single request.

        Takes the first request from the list of requests and returns it. If the list
        is empty, None is returned.

        Returns:
            Response: If a new request is available a Request object is returned,
                      otherwise None is returned.
        """
        pickled_request = self._connection.connection.lpop(self._request_key)
        return pickle.loads(pickled_request) if pickled_request is not None else None