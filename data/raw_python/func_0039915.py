def _get_dummy_request(base_url, user):
    """
    Create a dummy request.
    Use the ``base_url``, so code can use ``request.build_absolute_uri()`` to create absolute URLs.
    """
    split_url = urlsplit(base_url)
    is_secure = split_url[0] == 'https'
    dummy_request = RequestFactory(HTTP_HOST=split_url[1]).get('/', secure=is_secure)
    dummy_request.is_secure = lambda: is_secure
    dummy_request.user = user or AnonymousUser()
    dummy_request.site = None  # Workaround for wagtail.contrib.settings.context_processors
    return dummy_request