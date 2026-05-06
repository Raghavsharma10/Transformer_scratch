def authorization_url(self, response, state=""):
        """
        Return the authorization url that's needed to authorize as a user.

        :param response: Can be either code or pin. If it's code the user will
            be redirected to your redirect url with the code as a get parameter
            after authorizing your application. If it's pin then after
            authorizing your application, the user will instead be shown a pin
            on Imgurs website. Both code and pin are used to get an
            access_token and refresh token with the exchange_code and
            exchange_pin functions respectively.
        :param state: This optional parameter indicates any state which may be
            useful to your application upon receipt of the response. Imgur
            round-trips this parameter, so your application receives the same
            value it sent. Possible uses include redirecting the user to the
            correct resource in your site, nonces, and
            cross-site-request-forgery mitigations.
        """
        return AUTHORIZE_URL.format(self._base_url, self.client_id, response, state)