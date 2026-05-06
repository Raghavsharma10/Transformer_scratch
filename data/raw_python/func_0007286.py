def _get_response(self, method, endpoint, params=None):
        """
        Helper method to handle HTTP requests and catch API errors

        :param method: valid HTTP method
        :type method: str
        :param endpoint: API endpoint
        :type endpoint: str
        :param params: extra parameters passed with the request
        :type params: dict
        :returns: API response
        :rtype: Response
        """
        url = urljoin(self.api_url, endpoint)

        try:
            response = getattr(self._session, method)(url, params=params)

            # Check if Monzo API returned HTTP 401, which could mean that the
            # token is expired
            if response.status_code == 401:
                raise TokenExpiredError

        except TokenExpiredError:
            # For some reason 'requests-oauthlib' automatic token refreshing
            # doesn't work so we do it here semi-manually
            self._refresh_oath_token()

            self._session = OAuth2Session(
                client_id=self._client_id,
                token=self._token,
            )

            response = getattr(self._session, method)(url, params=params)

        if response.status_code != requests.codes.ok:
            raise MonzoAPIError(
                "Something went wrong: {}".format(response.json())
            )

        return response