def get_resp(url):
    """Get webpage response as an lxml.html.HtmlElement object."""
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        try:
            request = requests.get(url, headers=headers, proxies=get_proxies())
        except MissingSchema:
            url = add_protocol(url)
            request = requests.get(url, headers=headers, proxies=get_proxies())
        return lh.fromstring(request.text.encode('utf-8') if PY2 else request.text)
    except Exception:
        sys.stderr.write('Failed to retrieve {0}.\n'.format(url))
        raise