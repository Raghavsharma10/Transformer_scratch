def merge_xml(first_doc, second_doc):
    """Merges two XML documents.

    Args:
        first_doc (str): First XML document.  `second_doc` is merged into this
            document.
        second_doc (str): Second XML document.  It is merged into the first.

    Returns:
        XML Document: The merged document.

    Raises:
        None

    Example:
        >>> import pynos.utilities
        >>> import lxml
        >>> import xml
        >>> x = xml.etree.ElementTree.fromstring('<config />')
        >>> y = lxml.etree.fromstring('<config><hello /></config>')
        >>> x = pynos.utilities.merge_xml(x, y)
    """
    # Adapted from:
    # http://stackoverflow.com/questions/27258013/merge-two-xml-files-python
    # Maps each elements tag to the element from the first document
    if isinstance(first_doc, lxml.etree._Element):
        first_doc = ET.fromstring(lxml.etree.tostring(first_doc))
    if isinstance(second_doc, lxml.etree._Element):
        second_doc = ET.fromstring(lxml.etree.tostring(second_doc))
    mapping = {element.tag: element for element in first_doc}
    for element in second_doc:
        if not len(element):
            # Recursed fully.  This element has no children.
            try:
                # Update the first document's element's text
                mapping[element.tag].text = element.text
            except KeyError:
                # The element doesn't exist
                # add it to the mapping and the root document
                mapping[element.tag] = element
                first_doc.append(element)
        else:
            # This element has children.  Recurse.
            try:
                merge_xml(mapping[element.tag], element)
            except KeyError:
                # The element doesn't exist
                # add it to the mapping and the root document
                mapping[element.tag] = element
                first_doc.append(element)
    return lxml.etree.fromstring(ET.tostring(first_doc))