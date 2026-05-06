def to_raw_xml(source):
    """ Convert various representations of an XML structure to a normal XML string.

        Args:
            source -- The source object to be converted - ET.Element, dict or string.

        Returns:
            A rew xml string matching the source object.

    >>> to_raw_xml("<content/>")
    '<content/>'

    >>> to_raw_xml({'document': {'title': 'foo', 'list': [{'li':1}, {'li':2}]}})
    '<document><list><li>1</li><li>2</li></list><title>foo</title></document>'

    >>> to_raw_xml(ET.Element('root'))
    '<root/>'
    """
    if isinstance(source, basestring):
        return source
    elif hasattr(source, 'getiterator'):    # Element or ElementTree.
        return ET.tostring(source, encoding="utf-8")
    elif hasattr(source, 'keys'):   # Dict.
        xml_root = dict_to_etree(source)
        return ET.tostring(xml_root, encoding="utf-8")
    else:
        raise TypeError("Accepted representations of a document are string, dict and etree")