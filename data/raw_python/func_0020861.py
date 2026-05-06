def get_signature_string(params, secret):
    """
    Returns the unhashed signature string (secret + sorted list of param values) for an API call.
    @param params: dictionary values to generate signature string
    @param secret: secret string
    """
    str_list = [str(item) for item in extract_params(params)]
    str_list.sort()
    return (secret + ''.join(str_list)).encode('utf-8')