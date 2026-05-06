def pretty_xml(data):
    """Return a pretty formated xml
    """
    parsed_string = minidom.parseString(data.decode('utf-8'))
    return parsed_string.toprettyxml(indent='\t', encoding='utf-8')