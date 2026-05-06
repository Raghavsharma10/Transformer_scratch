def load_json(json_data, decoder=None):
    """
    Load data from json string

    :param json_data: Stringified json object
    :type json_data: str | unicode
    :param decoder: Use custom json decoder
    :type decoder: T <= DateTimeDecoder
    :return: Json data
    :rtype: None | int | float | str | list | dict
    """
    if decoder is None:
        decoder = DateTimeDecoder
    return json.loads(json_data, object_hook=decoder.decode)