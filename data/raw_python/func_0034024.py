def media(request, path, hproPk=None):
    """Ask the server for a media and return it to the client browser. Forward cache headers"""

    if not settings.PIAPI_STANDALONE:
        (plugIt, baseURI, _) = getPlugItObject(hproPk)
    else:
        global plugIt, baseURI

    try:
        (media, contentType, cache_control) = plugIt.getMedia(path)
    except Exception as e:
        report_backend_error(request, e, 'meta', hproPk)
        return gen500(request, baseURI)

    if not media:  # No media returned
        raise Http404

    response = HttpResponse(media)
    response['Content-Type'] = contentType
    response['Content-Length'] = len(media)

    if cache_control:
        response['Cache-Control'] = cache_control

    return response