def auth_oauth2(self) -> dict:
        """
        Authorizes a user by OAuth2 to get access token
        """
        oauth_data = {
            'client_id': self._app_id,
            'display': 'mobile',
            'response_type': 'token',
            'scope': '+66560',
            'v': self.API_VERSION
        }
        response = self.post(self.OAUTH_URL, oauth_data)
        url_params = get_url_params(response.url, fragment=True)
        if 'access_token' in url_params:
            return url_params

        action_url = get_base_url(response.text)
        if action_url:
            response = self.get(action_url)
            return get_url_params(response.url)

        response_json = response.json()
        if 'error' in response_json['error']:
            exception_msg = '{}: {}'.format(response_json['error'],
                                            response_json['error_description'])
            raise VVKAuthException(exception_msg)