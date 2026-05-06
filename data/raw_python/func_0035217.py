def _build_authorization_request_url(
        self,
        response_type,
        state=None
    ):
        """Form URL to request an auth code or access token.
        Parameters
            response_type (str)
                Only 'code' (Authorization Code Grant) supported at this time
            state (str)
                Optional CSRF State token to send to server.
        Returns
            (str)
                The fully constructed authorization request URL.
        Raises
            LyftIllegalState (ApiError)
                Raised if response_type parameter is invalid.
        """
        if response_type not in auth.VALID_RESPONSE_TYPES:
            message = '{} is not a valid response type.'
            raise LyftIllegalState(message.format(response_type))

        args = OrderedDict([
            ('scope', ' '.join(self.scopes)),
            ('state', state),
            ('response_type', response_type),
            ('client_id', self.client_id),
        ])

        return build_url(auth.SERVER_HOST, auth.AUTHORIZE_PATH, args)