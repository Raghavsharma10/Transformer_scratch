def api_user(request, userPk, key=None, hproPk=None):
    """Return information about an user"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    if settings.PIAPI_STANDALONE:
        if not settings.PIAPI_REALUSERS:
            user = generate_user(pk=userPk)
            if user is None:
                return HttpResponseNotFound()
        else:
            user = get_object_or_404(DUser, pk=userPk)

        hproject = None
    else:
        from users.models import TechUser

        user = get_object_or_404(TechUser, pk=userPk)

        (_, _, hproject) = getPlugItObject(hproPk)

        user.ebuio_member = hproject.isMemberRead(user)
        user.ebuio_admin = hproject.isMemberWrite(user)
        user.subscription_labels = _get_subscription_labels(user, hproject)

    retour = {}

    # Append properties for the user data
    for prop in settings.PIAPI_USERDATA:
        if hasattr(user, prop):
            retour[prop] = getattr(user, prop)

    retour['id'] = str(retour['pk'])

    # Append the users organisation and access levels
    orgas = {}
    if user:
        limitedOrgas = []

        if hproject and hproject.plugItLimitOrgaJoinable:
            # Get List of Plugit Available Orgas first
            projectOrgaIds = hproject.plugItOrgaJoinable.order_by('name').values_list('pk', flat=True)
            for (orga, isAdmin) in user.getOrgas(distinct=True):
                if orga.pk in projectOrgaIds:
                    limitedOrgas.append((orga, isAdmin))
        elif hasattr(user, 'getOrgas'):
            limitedOrgas = user.getOrgas(distinct=True)

        # Create List
        orgas = [{'id': orga.pk, 'name': orga.name, 'codops': orga.ebu_codops, 'is_admin': isAdmin} for (orga, isAdmin)
                 in limitedOrgas]
    retour['orgas'] = orgas

    return HttpResponse(json.dumps(retour), content_type="application/json")