def reload(request):
    """Reload local requirements file."""
    refresh_packages.clean()
    refresh_packages.local()
    refresh_packages.remote()
    url = request.META.get('HTTP_REFERER')
    if url:
        return HttpResponseRedirect(url)
    else:
        return HttpResponse('Local requirements list has been reloaded.')