def pretty_xml(string_input, add_ns=False):
    """ pretty indent string_input """
    if add_ns:
        elem = "<foo "
        for key, value in DOC_CONTENT_ATTRIB.items():
            elem += ' %s="%s"' % (key, value)
        string_input = elem + ">" + string_input + "</foo>"
    doc = minidom.parseString(string_input)
    if add_ns:
        s1 = doc.childNodes[0].childNodes[0].toprettyxml("  ")
    else:
        s1 = doc.toprettyxml("  ")
    return s1