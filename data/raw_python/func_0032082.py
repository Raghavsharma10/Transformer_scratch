def parse_json_structure(string_item):
    """
    Given a raw representation of a json structure, returns the parsed corresponding data
    structure (``JsonRpcRequest`` or ``JsonRpcRequestBatch``)

    :param string_item:
    :return:
    """
    if not isinstance(string_item, str):
        raise TypeError("Expected str but got {} instead".format(type(string_item).__name__))

    try:
        item = json.loads(string_item)
    except json.JSONDecodeError:
        raise JsonRpcParseError()

    if isinstance(item, dict):
        return JsonRpcRequest.from_dict(item)
    elif isinstance(item, list):
        if len(item) == 0:
            raise JsonRpcInvalidRequestError()

        request_batch = JsonRpcRequestBatch([])
        for d in item:
            try:
                # handles the case of valid batch but with invalid
                # requests.
                if not isinstance(d, dict):
                    raise JsonRpcInvalidRequestError()
                # is dict, all fine
                parsed_entry = JsonRpcRequest.from_dict(d)
            except JsonRpcInvalidRequestError:
                parsed_entry = GenericResponse.INVALID_REQUEST
            request_batch.add_item(parsed_entry)
        return request_batch