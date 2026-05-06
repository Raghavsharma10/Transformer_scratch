def home(request, hproPk):
    """ Route the request to runURI if defined otherwise go to plugIt """

    if settings.PIAPI_STANDALONE:
        return main(request, '', hproPk)

    (plugIt, baseURI, hproject) = getPlugItObject(hproPk)
    if hproject.runURI:
        return HttpResponseRedirect(hproject.runURI)
    else:
        # Check if a custom url key is used
        if hasattr(hproject, 'plugItCustomUrlKey') and hproject.plugItCustomUrlKey:
            return HttpResponseRedirect(reverse('plugIt.views.main', args=(hproject.plugItCustomUrlKey, '')))

        return main(request, '', hproPk)