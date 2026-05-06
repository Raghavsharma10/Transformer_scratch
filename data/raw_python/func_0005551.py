def format_xml(xml_str: str, exceptions: bool=False):
    """
    Formats XML document as human-readable plain text.
    :param xml_str: str (Input XML str)
    :param exceptions: Raise exceptions on error
    :return: str (Formatted XML str)
    """
    try:
        import xml.dom.minidom
        return xml.dom.minidom.parseString(xml_str).toprettyxml()
    except Exception:
        if exceptions:
            raise
        return xml_str