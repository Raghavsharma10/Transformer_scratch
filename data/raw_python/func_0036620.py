def authenticate_credentials(self, request, access_token):
        """
        Authenticate the request, given the access token.
        """

        try:
            token = oauth2_provider.oauth2.models.AccessToken.objects.select_related('user')
            # provider_now switches to timezone aware datetime when
            # the oauth2_provider version supports to it.
            token = token.get(token=access_token, expires__gt=provider_now())
        except oauth2_provider.oauth2.models.AccessToken.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid token')

        user = token.user

        if not user.is_active:
            msg = 'User inactive or deleted: %s' % user.username
            raise exceptions.AuthenticationFailed(msg)

        return (user, token)