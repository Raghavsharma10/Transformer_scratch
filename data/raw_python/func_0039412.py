def prettyfy(response, format='json'):
    """A wrapper for pretty_json and pretty_xml
    """
    if format == 'json':
        return pretty_json(response.content)
    else:
        return pretty_xml(response.content)