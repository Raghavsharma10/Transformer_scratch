def parse(to_parse, ignore_whitespace_text_nodes=True, adapter=None):
    """
    Parse an XML document into an *xml4h*-wrapped DOM representation
    using an underlying XML library implementation.

    :param to_parse: an XML document file, document string, or the
        path to an XML file. If a string value is given that contains
        a ``<`` character it is treated as literal XML data, otherwise
        a string value is treated as a file path.
    :type to_parse: a file-like object or string
    :param bool ignore_whitespace_text_nodes: if ``True`` pure whitespace
        nodes are stripped from the parsed document, since these are
        usually noise introduced by XML docs serialized to be human-friendly.
    :param adapter: the *xml4h* implementation adapter class used to parse
        the document and to interact with the resulting nodes.
        If None, :attr:`best_adapter` will be used.
    :type adapter: adapter class or None

    :return: an :class:`xml4h.nodes.Document` node representing the
        parsed document.

    Delegates to an adapter's :meth:`~xml4h.impls.interface.parse_string` or
    :meth:`~xml4h.impls.interface.parse_file` implementation.
    """
    if adapter is None:
        adapter = best_adapter
    if isinstance(to_parse, basestring) and '<' in to_parse:
        return adapter.parse_string(to_parse, ignore_whitespace_text_nodes)
    else:
        return adapter.parse_file(to_parse, ignore_whitespace_text_nodes)