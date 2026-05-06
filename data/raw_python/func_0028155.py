def decode_json(json_input: Union[str, None] = None):
    """
    Simple wrapper of json.load and json.loads.

    If json_input is None the output is an empty dictionary.
    If the input is a string that ends in .json it is decoded using json.load.
    Otherwise it is decoded using json.loads.

    Parameters
    ----------
    json_input : str, None, optional
        input json object


    Returns
    -------
    Decoded json object

    >>> decode_json()
    {}
    >>> decode_json('{"flag":true}')
    {'flag': True}
    >>> decode_json('{"value":null}')
    {'value': None}
    """
    if json_input is None:
        return {}
    else:
        if isinstance(json_input, str) is False:
            raise TypeError()
        elif json_input[-5:] == ".json":
            with open(json_input) as f:
                decoded_json = json.load(f)
        else:
            decoded_json = json.loads(json_input)
    return decoded_json