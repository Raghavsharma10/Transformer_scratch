def get_token_from_authorization_code(self,
                                          authorization_code, redirect_uri):
        """Like `get_token`, but using an OAuth 2 authorization code.

        Use this method if you run a webserver that serves as an endpoint for
        the redirect URI. The webserver can retrieve the authorization code
        from the URL that is requested by ORCID.

        Parameters
        ----------
        :param redirect_uri: string
            The redirect uri of the institution.
        :param authorization_code: string
            The authorization code.

        Returns
        -------
        :returns: dict
            All data of the access token.  The access token itself is in the
            ``"access_token"`` key.
        """
        token_dict = {
            "client_id": self._key,
            "client_secret": self._secret,
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": redirect_uri,
        }
        response = requests.post(self._token_url, data=token_dict,
                                 headers={'Accept': 'application/json'},
                                 timeout=self._timeout)
        response.raise_for_status()
        if self.do_store_raw_response:
            self.raw_response = response
        return json.loads(response.text)