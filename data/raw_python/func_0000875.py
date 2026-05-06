def append_minidom_xml_to_elementtree_xml(parent, xml, recursive=False, attributes=None):
    """
    Recursively,
    Given an ElementTree.Element as parent, and a minidom instance as xml,
    append the tags and content from xml to parent
    Used primarily for adding a snippet of XML with <italic> tags
    attributes: a list of attribute names to copy
    """

    # Get the root tag name
    if recursive is False:
        tag_name = xml.documentElement.tagName
        node = xml.getElementsByTagName(tag_name)[0]
        new_elem = SubElement(parent, tag_name)
        if attributes:
            for attribute in attributes:
                if xml.documentElement.getAttribute(attribute):
                    new_elem.set(attribute, xml.documentElement.getAttribute(attribute))
    else:
        node = xml
        tag_name = node.tagName
        new_elem = parent

    i = 0
    for child_node in node.childNodes:
        if child_node.nodeName == '#text':
            if not new_elem.text and i <= 0:
                new_elem.text = child_node.nodeValue
            elif not new_elem.text and i > 0:
                new_elem_sub.tail = child_node.nodeValue
            else:
                new_elem_sub.tail = child_node.nodeValue

        elif child_node.childNodes is not None:
            new_elem_sub = SubElement(new_elem, child_node.tagName)
            new_elem_sub = append_minidom_xml_to_elementtree_xml(new_elem_sub, child_node,
                                                                 True, attributes)

        i = i + 1

    # Debug
    #encoding = 'utf-8'
    #rough_string = ElementTree.tostring(parent, encoding)
    #print rough_string

    return parent