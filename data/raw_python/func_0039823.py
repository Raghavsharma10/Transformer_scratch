def dict_to_etree(source, root_tag=None):
    """ Recursively load dict/list representation of an XML tree into an etree representation.

        Args:
            source -- A dictionary representing an XML document where identical children tags are
                    countained in a list.

        Keyword args:
            root_tag -- A parent tag in which to wrap the xml tree. If None, and the source dict
                    contains multiple root items, a list of etree's Elements will be returned.

        Returns:
            An ET.Element which is the root of an XML tree or a list of these.

    >>> dict_to_etree({'foo': 'lorem'}) #doctest: +ELLIPSIS
    <Element foo at 0x...>

    >>> dict_to_etree({'foo': 'lorem', 'bar': 'ipsum'}) #doctest: +ELLIPSIS
    [<Element foo at 0x...>, <Element bar at 0x...>]

    >>> ET.tostring(dict_to_etree({'document': {'item1': 'foo', 'item2': 'bar'}}))
    '<document><item2>bar</item2><item1>foo</item1></document>'

    >>> ET.tostring(dict_to_etree({'foo': 'baz'}, root_tag='document'))
    '<document><foo>baz</foo></document>'

    >>> ET.tostring(dict_to_etree({'title': 'foo', 'list': [{'li':1}, {'li':2}]}, root_tag='document'))
    '<document><list><li>1</li><li>2</li></list><title>foo</title></document>'
    """
    def dict_to_etree_recursive(source, parent):
        if hasattr(source, 'keys'):
            for key, value in source.iteritems():
                sub = ET.SubElement(parent, key)
                dict_to_etree_recursive(value, sub)
        elif isinstance(source, list):
            for element in source:
                dict_to_etree_recursive(element, parent)
        else:   # TODO: Add feature to include xml literals as special objects or a etree subtree
            parent.text = source

    if root_tag is None:
        if len(source) == 1:
            root_tag = source.keys()[0]
            source = source[root_tag]
        else:
            roots = []
            for tag, content in source.iteritems():
                root = ET.Element(tag)
                dict_to_etree_recursive(content, root)
                roots.append(root)
            return roots
    root = ET.Element(root_tag)
    dict_to_etree_recursive(source, root)
    return root