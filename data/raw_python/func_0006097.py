def anonymous_required(function=None, redirect_url=None):
    """
    Decorator requiring the current user to be anonymous (not logged in).
    """
    if not redirect_url:
        redirect_url = settings.LOGIN_REDIRECT_URL

    actual_decorator = user_passes_test(
        is_anonymous,
        login_url=redirect_url,
        redirect_field_name=None
    )

    if function:
        return actual_decorator(function)
    return actual_decorator