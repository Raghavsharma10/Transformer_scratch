def setUser(request):
    """In standalone mode, change the current user"""

    if not settings.PIAPI_STANDALONE or settings.PIAPI_REALUSERS:
        raise Http404

    request.session['plugit-standalone-usermode'] = request.GET.get('mode')

    return HttpResponse('')