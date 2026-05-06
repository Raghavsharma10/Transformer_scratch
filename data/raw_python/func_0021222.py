def authorize_url(self, duration, scopes, state, implicit=False):
        """Return the URL used out-of-band to grant access to your application.

        :param duration: Either ``permanent`` or ``temporary``. ``temporary``
            authorizations generate access tokens that last only 1
            hour. ``permanent`` authorizations additionally generate a refresh
            token that can be indefinitely used to generate new hour-long
            access tokens. Only ``temporary`` can be specified if ``implicit``
            is set to ``True``.
        :param scopes: A list of OAuth scopes to request authorization for.
        :param state: A string that will be reflected in the callback to
            ``redirect_uri``. This value should be temporarily unique to the
            client for whom the URL was generated for.
        :param implicit: (optional) Use the implicit grant flow (default:
            False). This flow is only available for UntrustedAuthenticators.

        """
        if self.redirect_uri is None:
            raise InvalidInvocation("redirect URI not provided")
        if implicit and not isinstance(self, UntrustedAuthenticator):
            raise InvalidInvocation(
                "Only UntrustedAuthentictor instances can "
                "use the implicit grant flow."
            )
        if implicit and duration != "temporary":
            raise InvalidInvocation(
                "The implicit grant flow only supports "
                "temporary access tokens."
            )

        params = {
            "client_id": self.client_id,
            "duration": duration,
            "redirect_uri": self.redirect_uri,
            "response_type": "token" if implicit else "code",
            "scope": " ".join(scopes),
            "state": state,
        }
        url = self._requestor.reddit_url + const.AUTHORIZATION_PATH
        request = Request("GET", url, params=params)
        return request.prepare().url