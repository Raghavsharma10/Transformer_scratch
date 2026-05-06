def parse_bewit(bewit):
    """
    Returns a `bewittuple` representing the parts of an encoded bewit string.
    This has the following named attributes:
        (id, expiration, mac, ext)

    :param bewit:
        A base64 encoded bewit string
    :type bewit: str
    """
    decoded_bewit = b64decode(bewit).decode('ascii')
    bewit_parts = decoded_bewit.split("\\")
    if len(bewit_parts) != 4:
        raise InvalidBewit('Expected 4 parts to bewit: %s' % decoded_bewit)
    return bewittuple(*bewit_parts)