def api_orgas(request, key=None, hproPk=None):
    """Return the list of organizations pk"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    list_orgas = []

    if settings.PIAPI_STANDALONE:
        list_orgas = [{'id': -1, 'name': 'EBU', 'codops': 'ZZEBU'},
                      {'id': -2, 'name': 'RTS', 'codops': 'CHRTS'},
                      {'id': -3, 'name': 'BBC', 'codops': 'GBEBU'},
                      {'id': -4, 'name': 'CNN', 'codops': 'USCNN'}]

    else:
        from organizations.models import Organization

        (_, _, hproject) = getPlugItObject(hproPk)

        if hproject and hproject.plugItLimitOrgaJoinable:
            orgas = hproject.plugItOrgaJoinable.order_by('name').all()
        else:
            orgas = Organization.objects.order_by('name').all()

        list_orgas = [{'id': orga.pk, 'name': orga.name, 'codops': orga.ebu_codops} for orga in orgas]

    retour = {'data': list_orgas}

    return HttpResponse(json.dumps(retour), content_type="application/json")