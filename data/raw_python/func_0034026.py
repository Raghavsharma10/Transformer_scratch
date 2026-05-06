def setOrga(request, hproPk=None):
    """Change the current orga"""

    if settings.PIAPI_STANDALONE:
        request.session['plugit-standalone-organame'] = request.GET.get('name')
        request.session['plugit-standalone-orgapk'] = request.GET.get('pk')
    else:

        (_, _, hproject) = getPlugItObject(hproPk)

        from organizations.models import Organization

        orga = get_object_or_404(Organization, pk=request.GET.get('orga'))

        if request.user.is_superuser or orga.isMember(request.user) or orga.isOwner(request.user):
            request.session['plugit-orgapk-' + str(hproject.pk)] = orga.pk

    return HttpResponse('')