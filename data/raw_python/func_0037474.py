def formdata_post(url, fields):
    """Send an HTTP request with a multipart/form-data body for the
    given URL and return the data returned by the server.
    """
    content_type, data = formdata_encode(fields)
    req = urllib2.Request(url, data)
    req.add_header('Content-Type', content_type)
    return urllib2.urlopen(req).read()