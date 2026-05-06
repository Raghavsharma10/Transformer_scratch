def split_line(msg: str) -> Tuple[str, str, List[str]]:
    """ Parse message according to rfc 2812 for routing """
    match = RE_IRCLINE.match(msg)
    if not match:
        raise ValueError("Invalid line")

    prefix = match.group("prefix") or ""
    command = match.group("command")
    params = (match.group("params") or "").split()
    message = match.group("message") or ""

    if message:
        params.append(message)

    return prefix, command, params