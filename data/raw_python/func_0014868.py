def url_params(request, except_params=None, as_is=False):
    """
    create string with GET-params of request

    usage example:
        c['sort_url'] = url_params(request, except_params=('sort',))
        ...
        <a href="{{ sort_url }}&sort=lab_number">Лабораторный номер</a>
    """
    if not request.GET:
        return ''
    params = []
    for key, value in request.GET.items():
        if except_params and key not in except_params:
            for v in request.GET.getlist(key):
                params.append('%s=%s' % (key, urlquote(v)))

    if as_is:
        str_params = '?' + '&'.join(params)
    else:
        str_params = '?' + '&'.join(params)
        str_params = urlquote(str_params)
    return mark_safe(str_params)