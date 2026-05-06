def generate_signature(secret, verb, url, nonce, data):
    """Generate a request signature compatible with BitMEX."""
    # Parse the url so we can remove the base and extract just the path.
    parsedURL = urllib.parse.urlparse(url)
    path = parsedURL.path
    if parsedURL.query:
        path = path + '?' + parsedURL.query

    # print "Computing HMAC: %s" % verb + path + str(nonce) + data
    message = bytes(verb + path + str(nonce) + data, 'utf-8')

    signature = hmac.new(secret.encode('utf-8'),
                         message,
                         digestmod=hashlib.sha256).hexdigest()
    return signature