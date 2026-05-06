def wafer_sso_url(context, sso_method):
    '''
    Return the correct URL to SSO with the given method.
    '''
    request = context.request
    url = reverse(getattr(views, '%s_login' % sso_method))
    if 'next' in request.GET:
        url += '?' + urlencode({'next': request.GET['next']})
    return url