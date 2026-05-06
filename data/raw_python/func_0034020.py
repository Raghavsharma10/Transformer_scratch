def get_current_orga(request, hproject, availableOrga):
    """Return the current orga to use"""

    # If nothing available return 404
    if len(availableOrga) == 0:
        raise Http404

    # Find the current orga
    currentOrgaId = request.session.get('plugit-orgapk-' + str(hproject.pk), None)

    # If we don't have a current one select the first available
    if currentOrgaId is None:
        (tmpOrga, _) = availableOrga[0]
        currentOrgaId = tmpOrga.pk
    else:
        # If the current Orga is not among the available ones reset to the first one
        availableOrgaIds = [o.pk for (o, r) in availableOrga]
        if currentOrgaId not in availableOrgaIds:
            (tmpOrga, _) = availableOrga[0]
            currentOrgaId = tmpOrga.pk

    from organizations.models import Organization

    realCurrentOrga = get_object_or_404(Organization, pk=currentOrgaId)

    return realCurrentOrga