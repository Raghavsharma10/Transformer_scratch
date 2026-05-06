def rename_attributes(element, attrs):
    """
    Renames the attributes of the element. Accepts the element and a dictionary
    of string values. The keys are the original names, and their values will be
    the altered names. This method treats all attributes as optional and will
    not fail on missing attributes.
    """
    for name in attrs.keys():
        if name not in element.attrib:
            continue
        else:
            element.attrib[attrs[name]] = element.attrib.pop(name)