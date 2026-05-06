def sso_redirect(request, scopes=list([]), return_to=None):
    """
    Generates a :model:`esi.CallbackRedirect` for the specified request.
    Redirects to EVE for login.
    Accepts a view or URL name as a redirect after SSO.
    """
    logger.debug("Initiating redirect of {0} session {1}".format(request.user, request.session.session_key[:5]))
    if isinstance(scopes, string_types):
        scopes = list([scopes])

    # ensure only one callback redirect model per session
    CallbackRedirect.objects.filter(session_key=request.session.session_key).delete()

    # ensure session installed in database
    if not request.session.exists(request.session.session_key):
        logger.debug("Creating new session before redirect.")
        request.session.create()

    if return_to:
        url = reverse(return_to)
    else:
        url = request.get_full_path()

    oauth = OAuth2Session(app_settings.ESI_SSO_CLIENT_ID, redirect_uri=app_settings.ESI_SSO_CALLBACK_URL, scope=scopes)
    redirect_url, state = oauth.authorization_url(app_settings.ESI_OAUTH_LOGIN_URL)

    CallbackRedirect.objects.create(session_key=request.session.session_key, state=state, url=url)
    logger.debug("Redirecting {0} session {1} to SSO. Callback will be redirected to {2}".format(request.user, request.session.session_key[:5], url))
    return redirect(redirect_url)