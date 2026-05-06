def _set_error_handler_callbacks(self, app):
        """
        Sets the error handler callbacks used by this extension
        """
        @app.errorhandler(NoAuthorizationError)
        def handle_no_auth_error(e):
            return self._unauthorized_callback(str(e))

        @app.errorhandler(InvalidHeaderError)
        def handle_invalid_header_error(e):
            return self._invalid_token_callback(str(e))

        @app.errorhandler(jwt.ExpiredSignatureError)
        def handle_expired_error(e):
            return self._expired_token_callback()

        @app.errorhandler(jwt.InvalidTokenError)
        def handle_invalid_token_error(e):
            return self._invalid_token_callback(str(e))