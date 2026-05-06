def find_iteration(
    url: Union[methods, str],
    itermode: Optional[str] = None,
    iterkey: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Find iteration mode and iteration key for a given :class:`slack.methods`

    Args:
        url: :class:`slack.methods` or string url
        itermode: Custom iteration mode
        iterkey: Custom iteration key

    Returns:
        :py:class:`tuple` (itermode, iterkey)
    """
    if isinstance(url, methods):
        if not itermode:
            itermode = url.value[1]
        if not iterkey:
            iterkey = url.value[2]

    if not iterkey or not itermode:
        raise ValueError("Iteration not supported for: {}".format(url))
    elif itermode not in ITERMODE:
        raise ValueError("Iteration not supported for: {}".format(itermode))

    return itermode, iterkey