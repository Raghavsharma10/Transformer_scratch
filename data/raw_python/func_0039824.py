def to_etree(source, root_tag=None):
    """ Convert various representations of an XML structure to a etree Element

        Args:
            source -- The source object to be converted - ET.Element\ElementTree, dict or string.

        Keyword args:
            root_tag -- A optional parent tag in which to wrap the xml tree if no root in dict representation.
                    See dict_to_etree()

        Returns:
            A etree Element matching the source object.

    >>> to_etree("<content/>")  #doctest: +ELLIPSIS
    <Element content at 0x...>

    >>> to_etree({'document': {'title': 'foo', 'list': [{'li':1}, {'li':2}]}})  #doctest: +ELLIPSIS
    <Element document at 0x...>

    >>> to_etree(ET.Element('root'))  #doctest: +ELLIPSIS
    <Element root at 0x...>
    """
    if hasattr(source, 'get_root'): #XXX:
        return source.get_root()
    elif isinstance(source, type(ET.Element('x'))):    #XXX: # cElementTree.Element isn't exposed directly
        return source
    elif isinstance(source, basestring):
        try:
            return ET.fromstring(source)
        except:
            raise XMLError(source)
    elif hasattr(source, 'keys'):   # Dict.
        return dict_to_etree(source, root_tag)
    else:
        raise XMLError(source)