def tokens_required(scopes='', new=False):
    """
    Decorator for views to request an ESI Token.
    Accepts required scopes as a space-delimited string
    or list of strings of scope names.
    Can require a new token to be retrieved by SSO.
    Returns a QueryDict of Tokens.
    """

    def decorator(view_func):
        @wraps(view_func, assigned=available_attrs(view_func))
        def _wrapped_view(request, *args, **kwargs):

            # if we're coming back from SSO for a new token, return it
            token = _check_callback(request)
            if token and new:
                tokens = Token.objects.filter(pk=token.pk)
                logger.debug("Returning new token.")
                return view_func(request, tokens, *args, **kwargs)

            if not new:
                # ensure user logged in to check existing tokens
                if not request.user.is_authenticated:
                    logger.debug(
                        "Session {0} is not logged in. Redirecting to login.".format(request.session.session_key[:5]))
                    from django.contrib.auth.views import redirect_to_login
                    return redirect_to_login(request.get_full_path())

                # collect tokens in db, check if still valid, return if any
                tokens = Token.objects.filter(user__pk=request.user.pk).require_scopes(scopes).require_valid()
                if tokens.exists():
                    logger.debug("Retrieved {0} tokens for {1} session {2}".format(tokens.count(), request.user,
                                                                                   request.session.session_key[:5]))
                    return view_func(request, tokens, *args, **kwargs)

            # trigger creation of new token via sso
            logger.debug("No tokens identified for {0} session {1}. Redirecting to SSO.".format(request.user, request.session.session_key[:5]))
            from esi.views import sso_redirect
            return sso_redirect(request, scopes=scopes)

        return _wrapped_view

    return decorator