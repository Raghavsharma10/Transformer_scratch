def _parse_status_code(response):
    """
    Return error string code if the response is an error, otherwise ``"OK"``
    """

    # This happens when a status response is expected
    if isinstance(response, string_types):
        return response

    # This happens when a list of structs are expected
    is_single_list = isinstance(response, list) and len(response) == 1
    if is_single_list and isinstance(response[0], string_types):
        return response[0]

    # This happens when a struct of any kind is returned
    return "OK"