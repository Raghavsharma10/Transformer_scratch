def decode_body(headers: MutableMapping, body: bytes) -> dict:
    """
    Decode the response body

    For 'application/json' content-type load the body as a dictionary

    Args:
        headers: Response headers
        body: Response body

    Returns:
        decoded body
    """

    type_, encoding = parse_content_type(headers)
    decoded_body = body.decode(encoding)

    # There is one api that just returns `ok` instead of json. In order to have a consistent API we decided to modify the returned payload into a dict.
    if type_ == "application/json":
        payload = json.loads(decoded_body)
    else:
        if decoded_body == "ok":
            payload = {"ok": True}
        else:
            payload = {"ok": False, "data": decoded_body}

    return payload