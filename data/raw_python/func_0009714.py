def scrape(url, params=None, user_agent=None):
    '''
    Scrape a URL optionally with parameters.
    This is effectively a wrapper around urllib2.urlopen.
    '''

    headers = {}

    if user_agent:
        headers['User-Agent'] = user_agent

    data = params and six.moves.urllib.parse.urlencode(params) or None
    req = six.moves.urllib.request.Request(url, data=data, headers=headers)
    f = six.moves.urllib.request.urlopen(req)

    text = f.read()
    f.close()

    return text