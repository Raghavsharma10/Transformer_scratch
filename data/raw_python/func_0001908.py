def load_file(path):
    """
    Load file

    :param path: Path to file
    :type path: str | unicode
    :return: Loaded data
    :rtype: None | int | float | str | unicode | list | dict
    :raises IOError: If file not found or error accessing file
    """
    res = {}

    if not path:
        IOError("No path specified to save")

    if not os.path.isfile(path):
        raise IOError("File not found {}".format(path))

    try:
        with io.open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                res = load_json_file(f)
            elif path.endswith(".yaml") or path.endswith(".yml"):
                res = load_yaml_file(f)
    except IOError:
        raise
    except Exception as e:
        raise IOError(e)
    return res