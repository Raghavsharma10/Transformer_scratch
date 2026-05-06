def post_silence_request(self, kwargs):
        """
        Create a silence entry.
        """
        self._request('POST', '/silenced', data=json.dumps(kwargs))
        return True