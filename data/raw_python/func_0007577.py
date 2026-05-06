def sobject_to_json(obj, key_to_lower=False):
    """
    Converts a suds object to a JSON string.
    :param obj: suds object
    :param key_to_lower: If set, changes index key name to lower case.
    :return: json object
    """
    import json
    data = sobject_to_dict(obj, key_to_lower=key_to_lower, json_serialize=True)
    return json.dumps(data)