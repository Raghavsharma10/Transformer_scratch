def requires_auth(func):
    """Decorator to note which object methods require authorization."""
    @wraps(func)
    def auth_wrapper(self, *args, **kwargs):
        auth = False
        if hasattr(self, '_session'):
            auth = (self._session.auth or
                    self._session.headers.get('Authorization'))

        if auth:
            return func(self, *args, **kwargs)
        else:
            from .models import GitHubError
            # Mock a 401 response
            r = generate_fake_error_response(
                '{"message": "Requires authentication"}'
            )
            raise GitHubError(r)
    return auth_wrapper