def set_grant_type(self, grant_type = 'client_credentials', api_key=None, api_secret=None, scope=None, info=None):

        """
        Grant types:
         - token:
            An authorization is requested to the end-user by redirecting it to an authorization page hosted
            on Dailymotion. Once authorized, a refresh token is requested by the API client to the token
            server and stored in the end-user's cookie (or other storage technique implemented by subclasses).
            The refresh token is then used to request time limited access token to the token server.

        - none / client_credentials:
            This grant type is a 2 legs authentication: it doesn't allow to act on behalf of another user.
            With this grant type, all API requests will be performed with the user identity of the API key owner.

        - password:
            This grant type allows to authenticate end-user by directly providing its credentials.
            This profile is highly discouraged for web-server workflows. If used, the username and password
            MUST NOT be stored by the client.
        """

        self.access_token = None

        if api_key and api_secret:
            self._grant_info['key'] = api_key
            self._grant_info['secret'] = api_secret
        else:
            raise DailymotionClientError('Missing API key/secret')

        if isinstance(info, dict):
            self._grant_info.update(info)
        else:
            info = {}

        if self._session_store_enabled and isinstance(info, dict) and info.get('username') is not None:
            self._session_store.set_user(info.get('username'))

        if grant_type in ('authorization', 'token'):
            grant_type = 'authorization'
            if 'redirect_uri' not in info:
                raise DailymotionClientError('Missing redirect_uri in grant info for token grant type.')
        elif grant_type in ('client_credentials', 'none'):
            grant_type = 'client_credentials'
        elif grant_type == 'password':
            if 'username' not in info or 'password' not in info:
                raise DailymotionClientError('Missing username or password in grant info for password grant type.')

        self._grant_type = grant_type

        if scope:
            if not isinstance(scope, (list, tuple)):
                raise DailymotionClientError('Invalid scope type: must be a list of valid scopes')
            self._grant_info['scope'] = scope