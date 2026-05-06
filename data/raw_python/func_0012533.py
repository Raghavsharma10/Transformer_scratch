def load_json(json_file, **kwargs):
    """
    Open and load data from a JSON file

    .. code:: python

        reusables.load_json("example.json")
        # {u'key_1': u'val_1', u'key_for_dict': {u'sub_dict_key': 8}}

    :param json_file: Path to JSON file as string
    :param kwargs: Additional arguments for the json.load command
    :return: Dictionary
    """
    with open(json_file) as f:
        return json.load(f, **kwargs)