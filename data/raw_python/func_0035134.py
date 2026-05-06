def _parse_response(resp):
    """ Get xmlrpc response from scgi response
    """
    # Assume they care for standards and send us CRLF (not just LF)
    try:
        headers, payload = resp.split("\r\n\r\n", 1)
    except (TypeError, ValueError) as exc:
        raise SCGIException("No header delimiter in SCGI response of length %d (%s)" % (len(resp), exc,))
    headers = _parse_headers(headers)

    clen = headers.get("Content-Length")
    if clen is not None:
        # Check length, just in case the transport is bogus
        assert len(payload) == int(clen)

    return payload, headers