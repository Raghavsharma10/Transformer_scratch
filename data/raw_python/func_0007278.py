def get_req(url):
    """simple get request"""
    request = urllib.request.Request(url)
    request.add_header('Authorization', 'token %s' % API_TOKEN)
    try:
        response = urllib.request.urlopen(request).read().decode('utf-8')
        return response
    except urllib.error.HTTPError:
        exception()
        sys.exit(0)