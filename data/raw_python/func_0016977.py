def requires_app_credentials(func):
    """Require client_id and client_secret to be associated.

    This is used to note and enforce which methods require a client_id and
    client_secret to be used.

    """
    @wraps(func)
    def auth_wrapper(self, *args, **kwargs):
        client_id, client_secret = self._session.retrieve_client_credentials()
        if client_id and client_secret:
            return func(self, *args, **kwargs)
        else:
            from .models import GitHubError
            # Mock a 401 response
            r = generate_fake_error_response(
                '{"message": "Requires username/password authentication"}'
            )
            raise GitHubError(r)

    return auth_wrapper