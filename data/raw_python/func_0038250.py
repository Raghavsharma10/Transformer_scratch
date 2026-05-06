def access_required(config=None):
    """
    Authenticates a HTTP method handler based on a custom set of arguments
    """

    def _access_required(http_method_handler):

        def secure_http_method_handler(self, *args, **kwargs):

            # authentication context must be set
            if not self.__provider_config__.authentication:
                _message = "Service available to authenticated users only, no auth context provider set in handler"
                authentication_error = prestans.exception.AuthenticationError(_message)
                authentication_error.request = self.request
                raise authentication_error

            # check for access by calling is_authorized_user
            if not self.__provider_config__.authentication.is_authorized_user(config):
                _message = "Service available to authorized users only"
                authorization_error = prestans.exception.AuthorizationError(_message)
                authorization_error.request = self.request
                raise authorization_error

            http_method_handler(self, *args, **kwargs)

        return wraps(http_method_handler)(secure_http_method_handler)

    return _access_required