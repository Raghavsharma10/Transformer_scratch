def save_file(path, data, readable=False):
    """
    Save to file

    :param path: File path to save
    :type path: str | unicode
    :param data: Data to save
    :type data: None | int | float | str | unicode | list | dict
    :param readable: Format file to be human readable (default: False)
    :type readable: bool
    :rtype: None
    :raises IOError: If empty path or error writing file
    """
    if not path:
        IOError("No path specified to save")

    try:
        with io.open(path, "w", encoding="utf-8") as f:
            if path.endswith(".json"):
                save_json_file(
                    f,
                    data,
                    pretty=readable,
                    compact=(not readable),
                    sort=True
                )
            elif path.endswith(".yaml") or path.endswith(".yml"):
                save_yaml_file(f, data)
    except IOError:
        raise
    except Exception as e:
        raise IOError(e)