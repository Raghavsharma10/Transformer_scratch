def restore(self, request):
        """ Push the request back onto the queue.

        Args:
            request (Request): Reference to a request object that should be pushed back
                               onto the request queue.
        """
        self._connection.connection.rpush(self._request_key, pickle.dumps(request))