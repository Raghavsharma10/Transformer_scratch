def _parse_command_response(response):
    """Parse an SCI command response into ElementTree XML

    This is a helper method that takes a Requests Response object
    of an SCI command response and will parse it into an ElementTree Element
    representing the root of the XML response.

    :param response: The requests response object
    :return: An ElementTree Element that is the root of the response XML
    :raises ResponseParseError: If the response XML is not well formed
    """
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        raise ResponseParseError(
            "Unexpected response format, could not parse XML. Response: {}".format(response.text))

    return root