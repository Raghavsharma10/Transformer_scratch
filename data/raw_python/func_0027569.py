def build(tagname_or_element, ns_uri=None, adapter=None):
    """
    Return a :class:`~xml4h.builder.Builder` that represents an element in
    a new or existing XML DOM and provides "chainable" methods focussed
    specifically on adding XML content.

    :param tagname_or_element: a string name for the root node of a
        new XML document, or an :class:`~xml4h.nodes.Element` node in an
        existing document.
    :type tagname_or_element: string or :class:`~xml4h.nodes.Element` node
    :param ns_uri: a namespace URI to apply to the new root node. This
        argument has no effect this method is acting on an element.
    :type ns_uri: string or None
    :param adapter: the *xml4h* implementation adapter class used to
        interact with the document DOM nodes.
        If None, :attr:`best_adapter` will be used.
    :type adapter: adapter class or None

    :return: a :class:`~xml4h.builder.Builder` instance that represents an
        :class:`~xml4h.nodes.Element` node in an XML DOM.
    """
    if adapter is None:
        adapter = best_adapter
    if isinstance(tagname_or_element, basestring):
        doc = adapter.create_document(
            tagname_or_element, ns_uri=ns_uri)
        element = doc.root
    elif isinstance(tagname_or_element, xml4h.nodes.Element):
        element = tagname_or_element
    else:
        raise xml4h.exceptions.IncorrectArgumentTypeException(
            tagname_or_element, [basestring, xml4h.nodes.Element])
    return Builder(element)