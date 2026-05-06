def delete_namespace(parsed_xml):
    """
    Identifies the namespace associated with the root node of a XML document
    and removes that names from the document.

    :param parsed_xml: lxml.Etree object.
    :return: Returns the sources document with the namespace removed.
    """
    if parsed_xml.getroot().tag.startswith('{'):
        root = parsed_xml.getroot().tag
        end_ns = root.find('}')
        remove_namespace(parsed_xml, root[1:end_ns])
    return parsed_xml