def api_orga(request, orgaPk, key=None, hproPk=None):
    """Return information about an organization"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    retour = {}

    if settings.PIAPI_STANDALONE:
        retour['pk'] = orgaPk
        if orgaPk == "-1":
            retour['name'] = 'EBU'
            retour['codops'] = 'zzebu'
        if orgaPk == "-2":
            retour['name'] = 'RTS'
            retour['codops'] = 'chrts'
        if orgaPk == "-3":
            retour['name'] = 'BBC'
            retour['codops'] = 'gbbbc'
        if orgaPk == "-4":
            retour['name'] = 'CNN'
            retour['codops'] = 'uscnn'

    else:
        from organizations.models import Organization

        orga = get_object_or_404(Organization, pk=orgaPk)

        retour['pk'] = orga.pk
        retour['name'] = orga.name
        retour['codops'] = orga.ebu_codops

    return HttpResponse(json.dumps(retour), content_type="application/json")