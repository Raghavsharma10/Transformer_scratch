def prepare_request(
    url: Union[str, methods],
    data: Optional[MutableMapping],
    headers: Optional[MutableMapping],
    global_headers: MutableMapping,
    token: str,
    as_json: Optional[bool] = None,
) -> Tuple[str, Union[str, MutableMapping], MutableMapping]:
    """
    Prepare outgoing request

    Create url, headers, add token to the body and if needed json encode it

    Args:
        url: :class:`slack.methods` item or string of url
        data: Outgoing data
        headers: Custom headers
        global_headers: Global headers
        token: Slack API token
        as_json: Post JSON to the slack API
    Returns:
        :py:class:`tuple` (url, body, headers)
    """

    if isinstance(url, methods):
        as_json = as_json or url.value[3]
        real_url = url.value[0]
    else:
        real_url = url
        as_json = False

    if not headers:
        headers = {**global_headers}
    else:
        headers = {**global_headers, **headers}

    payload: Optional[Union[str, MutableMapping]] = None
    if real_url.startswith(HOOK_URL) or (real_url.startswith(ROOT_URL) and as_json):
        payload, headers = _prepare_json_request(data, token, headers)
    elif real_url.startswith(ROOT_URL) and not as_json:
        payload = _prepare_form_encoded_request(data, token)
    else:
        real_url = ROOT_URL + real_url
        payload = _prepare_form_encoded_request(data, token)

    return real_url, payload, headers