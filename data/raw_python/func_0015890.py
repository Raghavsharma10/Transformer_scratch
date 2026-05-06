def _call(self, path, method, body=None, headers=None):
        """
        Wrapper around http.do_call that transforms some HTTPError into
        our own exceptions
        """
        try:
            resp = self.http.do_call(path, method, body, headers)
        except http.HTTPError as err:
            if err.status == 401:
                raise PermissionError('Insufficient permissions to query ' +
                    '%s with user %s :%s' % (path, self.user, err))
            raise
        return resp