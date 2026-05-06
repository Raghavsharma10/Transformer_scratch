def save_yaml_file(file, val):
    """
    Save data to yaml file

    :param file: Writable object or path to file
    :type file: FileIO | str | unicode
    :param val: Value or struct to save
    :type val: None | int | float | str | unicode | list | dict
    """
    opened = False

    if not hasattr(file, "write"):
        file = io.open(file, "w", encoding="utf-8")
        opened = True

    try:
        yaml.dump(val, file)
    finally:
        if opened:
            file.close()