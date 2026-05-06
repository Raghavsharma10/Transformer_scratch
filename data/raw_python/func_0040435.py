async def parse_response(response: ClientResponse, schema: dict) -> Any:
    """
    Validate and parse the BMA answer

    :param response: Response of aiohttp request
    :param schema: The expected response structure
    :return: the json data
    """
    try:
        data = await response.json()
        response.close()
        if schema is not None:
            jsonschema.validate(data, schema)
        return data
    except (TypeError, json.decoder.JSONDecodeError) as e:
        raise jsonschema.ValidationError("Could not parse json : {0}".format(str(e)))