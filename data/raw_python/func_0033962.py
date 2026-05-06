def get_ebuio_headers(request):
    """Return a dict with ebuio headers"""

    retour = {}

    for (key, value) in request.headers:
        if key.startswith('X-Plugit-'):
            key = key[9:]

            retour[key] = value

    return retour