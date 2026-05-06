def authorize(self, login, password, scopes=None, note='', note_url='',
                  client_id='', client_secret=''):
        """Obtain an authorization token from the GitHub API for the GitHub
        API.

        :param str login: (required)
        :param str password: (required)
        :param list scopes: (optional), areas you want this token to apply to,
            i.e., 'gist', 'user'
        :param str note: (optional), note about the authorization
        :param str note_url: (optional), url for the application
        :param str client_id: (optional), 20 character OAuth client key for
            which to create a token
        :param str client_secret: (optional), 40 character OAuth client secret
            for which to create the token
        :returns: :class:`Authorization <Authorization>`
        """
        json = None
        # TODO: Break this behaviour in 1.0 (Don't rely on self._session.auth)
        auth = None
        if self._session.auth:
            auth = self._session.auth
        elif login and password:
            auth = (login, password)

        if auth:
            url = self._build_url('authorizations')
            data = {'note': note, 'note_url': note_url,
                    'client_id': client_id, 'client_secret': client_secret}
            if scopes:
                data['scopes'] = scopes

            with self._session.temporary_basic_auth(*auth):
                json = self._json(self._post(url, data=data), 201)

        return Authorization(json, self) if json else None