def parse_error(text: str) -> Any:
    """
    Validate and parse the BMA answer from websocket

    :param text: the bma error
    :return: the json data
    """
    try:
        data = json.loads(text)
        jsonschema.validate(data, ERROR_SCHEMA)
    except (TypeError, json.decoder.JSONDecodeError) as e:
        raise jsonschema.ValidationError("Could not parse json : {0}".format(str(e)))

    return data