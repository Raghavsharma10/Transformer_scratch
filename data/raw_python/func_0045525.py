def complete(request, provider):
    """
        After first step of net authentication, we must validate the response.
        If everything is ok, we must do the following:
        1. If user is already authenticated:
            a. Try to login him again (strange variation but we must take it to account).
            b. Create new netID record in database.
            c. Merge authenticated account with newly created netID record.
            d. Redirect user to 'next' url stored in session.
        2. If user is anonymouse:
            a. Try to log him by identity and redirect to 'next' url.
            b. Create new  netID record in database.
            c. Try to automaticaly fill all extra fields with information returned form
            server. If successfull, login the user and redirect to 'next' url.
            d. Redirect user to extra page where he can fill all extra fields by hand.
    """
    # merge data from POST and GET methods
    data = request.GET.copy()
    data.update(request.POST)

    # In case of skipping begin step.
    if 'next_url' not in request.session:
        request.session['next_url'] = request.GET.get("next") or settings.LOGIN_REDIRECT_URL

    backend = get_backend(provider)
    response = backend.validate(request, data)

    if isinstance(response, HttpResponseRedirect):
        return response
    if request.user.is_authenticated():
        success = backend.login_user(request)
        backend.merge_accounts(request)
    else:
        success = backend.login_user(request)
        if not success and not settings.REGISTRATION_ALLOWED:
            messages.warning(request, lang.REGISTRATION_DISABLED)
            return redirect(settings.REGISTRATION_DISABLED_REDIRECT)
    if success:
        return redirect(request.session.pop('next_url', settings.LOGIN_REDIRECT_URL))
    return backend.complete(request, response)