def enable_selfalias(config, id_name):
    """
    This allows replacing id_name with "self".
    e.g. /users/joe/account == /users/self/account if joe is in the session
    as an authorized user
    """

    def context_found_subscriber(event):
        request = event.request
        user = getattr(request, 'user', None)
        if (request.matchdict and
                request.matchdict.get(id_name, None) == 'self' and
                user):
            request.matchdict[id_name] = user.username

    config.add_subscriber(context_found_subscriber, ContextFound)