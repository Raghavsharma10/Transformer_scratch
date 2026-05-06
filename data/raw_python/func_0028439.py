def _before_request(self):
        """
        The before_request callback.
        """
        g._request_id = str(uuid.uuid4())
        g._start_timestamp = time.time()