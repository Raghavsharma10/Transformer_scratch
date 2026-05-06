def requires_basic_auth(func):
    """Specific (basic) authentication decorator.

    This is used to note which object methods require username/password
    authorization and won't work with token based authorization.

    """
    @wraps(func)
    def auth_wrapper(self, *args, **kwargs):
        if hasattr(self, '_session') and self._session.auth:
            return func(self, *args, **kwargs)
        else:
            from .models import GitHubError
            # Mock a 401 response
            r = generate_fake_error_response(
                '{"message": "Requires username/password authentication"}'
            )
            raise GitHubError(r)
    return auth_wrapper