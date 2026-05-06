def get_salt(request):
    """
    return the user password salt.
    If the user doesn't exist return a pseudo salt.
    """
    try:
        username = request.POST["username"]
    except KeyError:
        # log.error("No 'username' in POST data?!?")
        return HttpResponseBadRequest()

    try:
        request.server_challenge = request.session[SERVER_CHALLENGE_KEY]
    except KeyError as err:
        # log.error("Can't get challenge from session: %s", err)
        return HttpResponseBadRequest()
    # log.debug("old challenge: %r", request.server_challenge)

    send_pseudo_salt=True

    form = UsernameForm(request, data=request.POST)
    if form.is_valid():
        send_pseudo_salt=False

        user_profile = form.user_profile
        init_pbkdf2_salt = user_profile.init_pbkdf2_salt
        if not init_pbkdf2_salt:
            # log.error("No init_pbkdf2_salt set in user profile!")
            send_pseudo_salt=True

        if len(init_pbkdf2_salt)!=app_settings.PBKDF2_SALT_LENGTH:
            # log.error("Salt for user %r has wrong length: %r" % (request.POST["username"], init_pbkdf2_salt))
            send_pseudo_salt=True
    # else:
        # log.error("Salt Form is not valid: %r", form.errors)

    if send_pseudo_salt:
        # log.debug("\nUse pseudo salt!!!")
        init_pbkdf2_salt = crypt.get_pseudo_salt(app_settings.PBKDF2_SALT_LENGTH, username)

    response = HttpResponse(init_pbkdf2_salt, content_type="text/plain")

    if not send_pseudo_salt:
        response.add_duration=True # collect duration time in @TimingAttackPreventer

    # log.debug("\nsend init_pbkdf2_salt %r to client.", init_pbkdf2_salt)
    return response