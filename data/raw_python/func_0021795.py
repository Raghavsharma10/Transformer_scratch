def decode_response(status: int, headers: MutableMapping, body: bytes) -> dict:
    """
    Decode incoming response

    Args:
        status: Response status
        headers: Response headers
        body: Response body

    Returns:
        Response data
    """
    data = decode_body(headers, body)
    raise_for_status(status, headers, data)
    raise_for_api_error(headers, data)

    return data