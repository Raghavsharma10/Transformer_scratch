def validate(in_, options=None):
    """
    Validate objects from JSON data in a textual stream.

    :param in_: A textual stream of JSON data.
    :param options: Validation options
    :return: An ObjectValidationResults instance, or a list of such.
    """
    obj_json = json.load(in_)

    results = validate_parsed_json(obj_json, options)

    return results