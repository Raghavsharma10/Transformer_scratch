def api_get_project_members(request, key=None, hproPk=True):
    """Return the list of project members"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    if settings.PIAPI_STANDALONE:
        if not settings.PIAPI_REALUSERS:
            users = [generate_user(pk="-1"), generate_user(pk="-2"), generate_user(pk="-3")]
        else:
            users = DUser.object.all()
    else:

        (_, _, hproject) = getPlugItObject(hproPk)

        users = []

        for u in hproject.getMembers():
            u.ebuio_member = True
            u.ebuio_admin = hproject.isMemberWrite(u)
            u.subscription_labels = _get_subscription_labels(u, hproject)
            users.append(u)

    liste = []

    for u in users:

        retour = {}

        for prop in settings.PIAPI_USERDATA:
            if hasattr(u, prop):
                retour[prop] = getattr(u, prop)

        retour['id'] = str(retour['pk'])

        liste.append(retour)

    return HttpResponse(json.dumps({'members': liste}), content_type="application/json")