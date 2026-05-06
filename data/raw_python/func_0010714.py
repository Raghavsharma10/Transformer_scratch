def send(self, response):
        """ Send a response back to the client that issued a request.

        Args:
            response (Response): Reference to the response object that should be sent.
        """
        self._connection.connection.set('{}:{}'.format(SIGNAL_REDIS_PREFIX, response.uid),
                                        pickle.dumps(response))