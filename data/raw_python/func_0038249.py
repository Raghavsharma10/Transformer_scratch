def role_required(role_name=None):
    """
    Authenticates a HTTP method handler based on a provided role

    With a little help from Peter Cole's Blog
    http://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
    """

    def _role_required(http_method_handler):

        @wraps(http_method_handler)
        def secure_http_method_handler(self, *args, **kwargs):

            # role name must be provided
            if role_name is None:
                _message = "Role name must be provided"
                authorization_error = prestans.exception.AuthorizationError(_message)
                authorization_error.request = self.request
                raise authorization_error

            # authentication context must be set
            if not self.__provider_config__.authentication:
                _message = "Service available to authenticated users only, no auth context provider set in handler"
                authentication_error = prestans.exception.AuthenticationError(_message)
                authentication_error.request = self.request
                raise authentication_error

            # check for the role by calling current_user_has_role
            if not self.__provider_config__.authentication.current_user_has_role(role_name):
                authorization_error = prestans.exception.AuthorizationError(role_name)
                authorization_error.request = self.request
                raise authorization_error

            http_method_handler(self, *args, **kwargs)

        return wraps(http_method_handler)(secure_http_method_handler)

    return _role_required