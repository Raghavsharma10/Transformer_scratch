def save_json(val, pretty=False, sort=True, encoder=None):
    """
    Save data to json string

    :param val: Value or struct to save
    :type val: None | int | float | str | list | dict
    :param pretty: Format data to be readable (default: False)
                    otherwise going to be compact
    :type pretty: bool
    :param sort: Sort keys (default: True)
    :type sort: bool
    :param encoder: Use custom json encoder
    :type encoder: T <= DateTimeEncoder
    :return: The jsonified string
    :rtype: str | unicode
    """
    if encoder is None:
        encoder = DateTimeEncoder
    if pretty:
        data = json.dumps(
            val,
            indent=4,
            separators=(',', ': '),
            sort_keys=sort,
            cls=encoder
        )
    else:
        data = json.dumps(
            val,
            separators=(',', ':'),
            sort_keys=sort,
            cls=encoder
        )
    if not sys.version_info > (3, 0) and isinstance(data, str):
        data = data.decode("utf-8")
    return data