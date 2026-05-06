def parse_text(text: str, schema: dict) -> Any:
    """
    Validate and parse the BMA answer from websocket

    :param text: the bma answer
    :param schema: dict for jsonschema
    :return: the json data
    """
    try:
        data = json.loads(text)
        jsonschema.validate(data, schema)
    except (TypeError, json.decoder.JSONDecodeError):
        raise jsonschema.ValidationError("Could not parse json")

    return data