def post_check_request(self, check, subscribers):
        """
        Issues a check execution request.
        """
        data = {
            'check': check,
            'subscribers': [subscribers]
        }
        self._request('POST', '/request', data=json.dumps(data))
        return True