def email_login(request, *, email, **kwargs):
    """
    Given a request, an email and optionally some additional data, ensure that
    a user with the email address exists, and authenticate & login them right
    away if the user is active.

    Returns a tuple consisting of ``(user, created)`` upon success or ``(None,
    None)`` when authentication fails.
    """
    _u, created = auth.get_user_model()._default_manager.get_or_create(email=email)
    user = auth.authenticate(request, email=email)
    if user and user.is_active:  # The is_active check is possibly redundant.
        auth.login(request, user)
        return user, created
    return None, None