def sso(user, desired_username, name, email, profile_fields=None):
    """
    Create a user, if the provided `user` is None, from the parameters.
    Then log the user in, and return it.
    """
    if not user:
        if not settings.REGISTRATION_OPEN:
            raise SSOError('Account registration is closed')
        user = _create_desired_user(desired_username)
        _configure_user(user, name, email, profile_fields)

    if not user.is_active:
        raise SSOError('Account disabled')

    # login() expects the logging in backend to be set on the user.
    # We are bypassing login, so fake it.
    user.backend = settings.AUTHENTICATION_BACKENDS[0]
    return user