def create(self, token, email, password):
        """
        Create a new token

        :param Token token: Token instance to create.
        :param str email: Email address of the Koordinates user account.
        :param str password: Koordinates user account password.
        """
        target_url = self.client.get_url('TOKEN', 'POST', 'create')
        post_data = {
            'grant_type': 'password',
            'username': email,
            'password': password,
            'name': token.name,
        }
        if getattr(token, 'scope', None):
            post_data['scope'] = token.scope
        if getattr(token, 'expires_at', None):
            post_data['expires_at'] = token.expires_at
        if getattr(token, 'referrers', None):
            post_data['referrers'] = token.referrers

        r = self.client._raw_request('POST', target_url, json=post_data, headers={'Content-type': 'application/json'})
        return self.create_from_result(r.json())