def open_url(url, httpuser=None, httppassword=None, method=None):
    """
    Open a URL using an opener that will simulate a browser user-agent
    url: The URL
    httpuser, httppassword: HTTP authentication credentials (either both or
      neither must be provided)
    method: The HTTP method

    Caller is reponsible for calling close() on the returned object
    """
    if os.getenv('OMEGO_SSL_NO_VERIFY') == '1':
        # This needs to come first to override the default HTTPS handler
        log.debug('OMEGO_SSL_NO_VERIFY=1')
        try:
            sslctx = ssl.create_default_context()
        except Exception as e:
            log.error('Failed to create Default SSL context: %s' % e)
            raise Stop(
                'Failed to create Default SSL context, OMEGO_SSL_NO_VERIFY '
                'is not supported on older versions of Python')
        sslctx.check_hostname = False
        sslctx.verify_mode = ssl.CERT_NONE
        opener = urllib2.build_opener(urllib2.HTTPSHandler(context=sslctx))
    else:
        opener = urllib2.build_opener()

    if 'USER_AGENT' in os.environ:
        opener.addheaders = [('User-agent', os.environ.get('USER_AGENT'))]
        log.debug('Setting user-agent: %s', os.environ.get('USER_AGENT'))

    if httpuser and httppassword:
        mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, url, httpuser, httppassword)
        log.debug('Enabling HTTP authentication')
        opener.add_handler(urllib2.HTTPBasicAuthHandler(mgr))
        opener.add_handler(urllib2.HTTPDigestAuthHandler(mgr))
    elif httpuser or httppassword:
        raise FileException(
            'httpuser and httppassword must be used together', url)

    # Override method http://stackoverflow.com/a/4421485
    req = urllib2.Request(url)
    if method:
        req.get_method = lambda: method

    return opener.open(req)