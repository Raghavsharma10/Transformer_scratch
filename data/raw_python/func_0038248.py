def login_required(http_method_handler):
    """
    provides a decorator for RESTRequestHandler methods to check for authenticated users

    RESTRequestHandler subclass must have a auth_context instance, refer to prestans.auth
    for the parent class definition.

    If decorator is used and no auth_context is provided the client will be denied access.

    Handler will return a 401 Unauthorized if the user is not logged in, the service does
    not redirect to login handler page, this is the client's responsibility.

    auth_context_handler instance provides a message called get_current_user, use this
    to obtain a reference to an authenticated user profile.

    If all goes well, the original handler definition is executed.
    """

    @wraps(http_method_handler)
    def secure_http_method_handler(self, *args, **kwargs):

        if not self.__provider_config__.authentication:
            _message = "Service available to authenticated users only, no auth context provider set in handler"
            authentication_error = prestans.exception.AuthenticationError(_message)
            authentication_error.request = self.request
            raise authentication_error

        if not self.__provider_config__.authentication.is_authenticated_user():
            authentication_error = prestans.exception.AuthenticationError()
            authentication_error.request = self.request
            raise authentication_error

        http_method_handler(self, *args, **kwargs)

    return secure_http_method_handler