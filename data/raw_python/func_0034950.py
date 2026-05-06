def _build_headers(self, method, auth_session):
        """Create headers for the request.
        Parameters
            method (str)
                HTTP method (e.g. 'POST').
            auth_session (Session)
                The Session object containing OAuth 2.0 credentials.
        Returns
            headers (dict)
                Dictionary of access headers to attach to request.
        Raises
            LyftIllegalState (ApiError)
                Raised if headers are invalid.
        """
        token_type = auth_session.token_type

        token = auth_session.oauth2credential.access_token

        if not self._authorization_headers_valid(token_type, token):
            message = 'Invalid token_type or token.'
            raise LyftIllegalState(message)

        headers = {
            'Authorization': ' '.join([token_type, token]),
        }

        if method in http.BODY_METHODS:
            headers.update(http.DEFAULT_CONTENT_HEADERS)

        return headers