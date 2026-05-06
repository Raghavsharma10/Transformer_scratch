def index(request):
    """
    Redirects to the default wiki index name.
    """
    kwargs = {'slug': getattr(settings, 'WAKAWAKA_DEFAULT_INDEX', 'WikiIndex')}
    redirect_to = reverse('wakawaka_page', kwargs=kwargs)
    return HttpResponseRedirect(redirect_to)