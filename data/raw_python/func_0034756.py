def parse_digest_response(digest_response_string):
    '''
    Parse the parameters of a Digest response. The input is a comma separated list of
    token=(token|quoted-string). See RFCs 2616 and 2617 for details.

    Known issue: this implementation will fail if there are commas embedded in quoted-strings.
    '''

    parts = parse_parts(digest_response_string, defaults={'algorithm': 'MD5'})
    if not _check_required_parts(parts, _REQUIRED_DIGEST_RESPONSE_PARTS):
        return None

    if not parts['nc'] or [c for c in parts['nc'] if not c in '0123456789abcdefABCDEF']:
        return None
    parts['nc'] = int(parts['nc'], 16)

    digest_response = _build_object_from_parts(parts, _REQUIRED_DIGEST_RESPONSE_PARTS)
    if ('MD5', 'auth') != (digest_response.algorithm, digest_response.qop):
        return None

    return digest_response