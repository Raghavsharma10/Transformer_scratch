def receive_callback(request):
    """
    Parses SSO callback, validates, retrieves :model:`esi.Token`, and internally redirects to the target url.
    """
    logger.debug("Received callback for {0} session {1}".format(request.user, request.session.session_key[:5]))
    # make sure request has required parameters
    code = request.GET.get('code', None)
    state = request.GET.get('state', None)
    try:
        assert code
        assert state
    except AssertionError:
        logger.debug("Missing parameters for code exchange.")
        return HttpResponseBadRequest()

    callback = get_object_or_404(CallbackRedirect, state=state, session_key=request.session.session_key)
    token = Token.objects.create_from_request(request)
    callback.token = token
    callback.save()
    logger.debug(
        "Processed callback for {0} session {1}. Redirecting to {2}".format(request.user, request.session.session_key[:5], callback.url))
    return redirect(callback.url)