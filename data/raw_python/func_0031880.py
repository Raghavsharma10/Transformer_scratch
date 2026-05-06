def post_event(self, client, check):
        """
        Resolves an event. (delayed action)
        """
        self._request('POST', '/resolve',
                      data=json.dumps({'client': client, 'check': check}))
        return True