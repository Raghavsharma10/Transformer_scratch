def save_json_file(
        file, val,
        pretty=False, compact=True, sort=True, encoder=None
):
    """
    Save data to json file

    :param file: Writable object or path to file
    :type file: FileIO | str | unicode
    :param val: Value or struct to save
    :type val: None | int | float | str | list | dict
    :param pretty: Format data to be readable (default: False)
    :type pretty: bool
    :param compact: Format data to be compact (default: True)
    :type compact: bool
    :param sort: Sort keys (default: True)
    :type sort: bool
    :param encoder: Use custom json encoder
    :type encoder: T <= DateTimeEncoder
    :rtype: None
    """
    # TODO: make pretty/compact into one bool?
    if encoder is None:
        encoder = DateTimeEncoder
    opened = False

    if not hasattr(file, "write"):
        file = io.open(file, "w", encoding="utf-8")
        opened = True

    try:
        if pretty:
            data = json.dumps(
                val,
                indent=4,
                separators=(',', ': '),
                sort_keys=sort,
                cls=encoder
            )
        elif compact:
            data = json.dumps(
                val,
                separators=(',', ':'),
                sort_keys=sort,
                cls=encoder
            )
        else:
            data = json.dumps(val, sort_keys=sort, cls=encoder)
        if not sys.version_info > (3, 0) and isinstance(data, str):
            data = data.decode("utf-8")
        file.write(data)
    finally:
        if opened:
            file.close()