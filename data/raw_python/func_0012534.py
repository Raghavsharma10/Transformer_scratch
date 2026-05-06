def save_json(data, json_file, indent=4, **kwargs):
    """
    Takes a dictionary and saves it to a file as JSON

    .. code:: python

        my_dict = {"key_1": "val_1",
                   "key_for_dict": {"sub_dict_key": 8}}

        reusables.save_json(my_dict,"example.json")

    example.json

    .. code::

        {
            "key_1": "val_1",
            "key_for_dict": {
                "sub_dict_key": 8
            }
        }

    :param data: dictionary to save as JSON
    :param json_file: Path to save file location as str
    :param indent: Format the JSON file with so many numbers of spaces
    :param kwargs: Additional arguments for the json.dump command
    """
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)