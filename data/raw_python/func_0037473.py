def formdata_encode(fields):
    """Encode fields (a dict) as a multipart/form-data HTTP request
    payload. Returns a (content type, request body) pair.
    """
    BOUNDARY = '----form-data-boundary-ZmRkNzJkMjUtMjkyMC00'
    out = []
    for (key, value) in fields.items():
        out.append('--' + BOUNDARY)
        out.append('Content-Disposition: form-data; name="%s"' % key)
        out.append('')
        out.append(value)
    out.append('--' + BOUNDARY + '--')
    out.append('')
    body = '\r\n'.join(out)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body