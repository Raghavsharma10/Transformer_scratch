def check_proxy_setting():
    """
    If the environmental variable 'HTTP_PROXY' is set, it will most likely be
    in one of these forms:

          proxyhost:8080
          http://proxyhost:8080

    urlllib2 requires the proxy URL to start with 'http://'
    This routine does that, and returns the transport for xmlrpc.
    """
    try:
        http_proxy = os.environ['HTTP_PROXY']
    except KeyError:
        return

    if not http_proxy.startswith('http://'):
        match = re.match('(http://)?([-_\.A-Za-z]+):(\d+)', http_proxy)
        #if not match:
        #    raise Exception('Proxy format not recognised: [%s]' % http_proxy)
        os.environ['HTTP_PROXY'] = 'http://%s:%s' % (match.group(2),
                match.group(3))
    return