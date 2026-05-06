def decode(encoded):
    # type: (bytes) -> bytes
    """
    Decode the result of querystringsafe_base64_encode or a regular base64.

    .. note ::
        As a regular base64 string does not contain dots, replacing dots with
        equal signs does basically noting to it. Also,
        base64.urlsafe_b64decode allows to decode both safe and unsafe base64.
        Therefore this function may also be used to decode the regular base64.

    :param (str, unicode) encoded: querystringsafe_base64 string or unicode
    :rtype: str, bytes
    :return: decoded string
    """
    padded_string = fill_padding(encoded)
    return urlsafe_b64decode(padded_string.replace(b'.', b'='))