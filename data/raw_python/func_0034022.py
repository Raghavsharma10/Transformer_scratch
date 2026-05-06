def get_current_session(request, hproPk):
    """Get the current session value"""

    retour = {}

    base_key = 'plugit_' + str(hproPk) + '_'

    for key, value in request.session.iteritems():
        if key.startswith(base_key):
            retour[key[len(base_key):]] = value

    return retour