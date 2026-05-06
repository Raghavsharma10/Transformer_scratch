def _is_api_key_correct(request):
    """Return whether the Geckoboard API key on the request is correct."""
    api_key = getattr(settings, 'GECKOBOARD_API_KEY', None)
    if api_key is None:
        return True
    auth = request.META.get('HTTP_AUTHORIZATION', '').split()
    if len(auth) == 2:
        if auth[0].lower() == b'basic':
            request_key = base64.b64decode(auth[1]).split(b':')[0]
            return request_key == api_key
    return False