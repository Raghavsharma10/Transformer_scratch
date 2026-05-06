def return_xml(element_tree):
    """Return an XML Element.

        Args:
            element_tree (Element): XML Element to be returned.  If sent as a
                ``str``, this function will attempt to convert it to an
                ``Element``.

        Returns:
            Element: An XML Element.

        Raises:
            TypeError: if `element_tree` is not of type ``Element`` and it
                cannot be converted from a ``str``.

        Examples:
            >>> import pynos.utilities
            >>> import xml.etree.ElementTree as ET
            >>> ele = pynos.utilities.return_xml(ET.Element('config'))
            >>> assert isinstance(ele, ET.Element)
            >>> ele = pynos.utilities.return_xml('<config />')
            >>> assert isinstance(ele, ET.Element)
            >>> ele = pynos.utilities.return_xml(
            ... ['hodor']) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            TypeError
    """
    if isinstance(element_tree, ET.Element):
        return element_tree
    try:
        return ET.fromstring(element_tree)
    except TypeError:
        raise TypeError('{} takes either {} or {} type.'
                        .format(repr(return_xml.__name__),
                                repr(str.__name__),
                                repr(ET.Element.__name__)))