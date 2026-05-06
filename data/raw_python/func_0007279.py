def geturl_req(url):
    """get request that returns 302"""
    request = urllib.request.Request(url)
    request.add_header('Authorization', 'token %s' % API_TOKEN)
    try:
        response_url = urllib.request.urlopen(request).geturl()
        return response_url
    except urllib.error.HTTPError:
        exception()
        sys.exit(0)