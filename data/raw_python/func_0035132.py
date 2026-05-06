def _encode_payload(data, headers=None):
    "Wrap data in an SCGI request."
    prolog = "CONTENT_LENGTH\0%d\0SCGI\x001\0" % len(data)
    if headers:
        prolog += _encode_headers(headers)

    return _encode_netstring(prolog) + data