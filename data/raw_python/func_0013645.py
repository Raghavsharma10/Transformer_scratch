def find_txt(xml_tree, path, default=''):
    """
    Extracts the text value from an XML tree, using XPath.
    In case of error, will return a default value.

    :param xml_tree: the XML Tree object. Assumed is <type 'lxml.etree._Element'>.
    :param path:     XPath to be applied, in order to extract the desired data.
    :param default:  Value to be returned in case of error.
    :return: a str value.
    """
    value = ''
    try:
        xpath_applied = xml_tree.xpath(path)  # will consider the first match only
        if len(xpath_applied) and xpath_applied[0] is not None:
            xpath_result = xpath_applied[0]
            if isinstance(xpath_result, type(xml_tree)):
                value = xpath_result.text.strip()
            else:
                value = xpath_result
    except Exception:  # in case of any exception, returns default
        value = default
    return py23_compat.text_type(value)