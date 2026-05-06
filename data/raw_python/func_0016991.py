def check_authorization(self, access_token):
        """OAuth applications can use this method to check token validity
        without hitting normal rate limits because of failed login attempts.
        If the token is valid, it will return True, otherwise it will return
        False.

        :returns: bool
        """
        p = self._session.params
        auth = (p.get('client_id'), p.get('client_secret'))
        if access_token and auth:
            url = self._build_url('applications', str(auth[0]), 'tokens',
                                  str(access_token))
            resp = self._get(url, auth=auth, params={
                'client_id': None, 'client_secret': None
            })
            return self._boolean(resp, 200, 404)
        return False