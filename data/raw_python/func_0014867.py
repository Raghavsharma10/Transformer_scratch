def url_path(request, base_url=None, is_full=False, *args, **kwargs):
    """
    join base_url and some GET-parameters to one; it could be absolute url optionally

    usage example:

        c['current_url'] = url_path(request, use_urllib=True, is_full=False)
        ...
        <a href="{{ current_url }}">Лабораторный номер</a>

    """
    if not base_url:
        base_url = request.path
        if is_full:
            protocol = 'https' if request.is_secure() else 'http'
            base_url = '%s://%s%s' % (protocol, request.get_host(), base_url)

    params = url_params(request, *args, **kwargs)
    url = '%s%s' % (base_url, params)
    return url