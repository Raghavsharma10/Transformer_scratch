def reset(self):
        """ Reset the connection

        """
        self._request = None
        self._response = None
        self._transaction_id = uuid.uuid4().hex