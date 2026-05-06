def prepare_iter_request(
    url: Union[methods, str],
    data: MutableMapping,
    *,
    iterkey: Optional[str] = None,
    itermode: Optional[str] = None,
    limit: int = 200,
    itervalue: Optional[Union[str, int]] = None,
) -> Tuple[MutableMapping, str, str]:
    """
    Prepare outgoing iteration request

    Args:
        url: :class:`slack.methods` item or string of url
        data: Outgoing data
        limit: Maximum number of results to return per call.
        iterkey: Key in response data to iterate over (required for url string).
        itermode: Iteration mode (required for url string) (one of `cursor`, `page` or `timeline`)
        itervalue: Value for current iteration (cursor hash, page or timestamp depending on the itermode)
    Returns:
        :py:class:`tuple` (data, iterkey, itermode)
    """
    itermode, iterkey = find_iteration(url, itermode, iterkey)

    if itermode == "cursor":
        data["limit"] = limit
        if itervalue:
            data["cursor"] = itervalue
    elif itermode == "page":
        data["count"] = limit
        if itervalue:
            data["page"] = itervalue
    elif itermode == "timeline":
        data["count"] = limit
        if itervalue:
            data["latest"] = itervalue

    return data, iterkey, itermode