def load_json_file(file, decoder=None):
    """
    Load data from json file

    :param file: Readable object or path to file
    :type file: FileIO | str
    :param decoder: Use custom json decoder
    :type decoder: T <= DateTimeDecoder
    :return: Json data
    :rtype: None | int | float | str | list | dict
    """
    if decoder is None:
        decoder = DateTimeDecoder
    if not hasattr(file, "read"):
        with io.open(file, "r", encoding="utf-8") as f:
            return json.load(f, object_hook=decoder.decode)
    return json.load(file, object_hook=decoder.decode)