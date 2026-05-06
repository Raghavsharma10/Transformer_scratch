def begin(request, provider):
    """
        Display authentication form. This is also the first step
        in registration. The actual login is in social_complete
        function below.
    """
    # store url to where user will be redirected
    request.session['next_url'] = request.GET.get("next") or settings.LOGIN_REDIRECT_URL

    # start the authentication process
    backend = get_backend(provider)
    return backend.begin(request, dict(request.REQUEST.items()))