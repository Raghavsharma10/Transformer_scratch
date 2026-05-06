def dumps(voevent, pretty_print=False, xml_declaration=True, encoding='UTF-8'):
    """Converts voevent to string.

    .. note:: Default encoding is UTF-8, in line with VOE2.0 schema.
        Declaring the encoding can cause diffs with the original loaded VOEvent,
        but I think it's probably the right thing to do (and lxml doesn't
        really give you a choice anyway).

    Args:
        voevent (:class:`Voevent`): Root node of the VOevent etree.
        pretty_print (bool): indent the output for improved human-legibility
            when possible. See also:
            http://lxml.de/FAQ.html#why-doesn-t-the-pretty-print-option-reformat-my-xml-output
        xml_declaration (bool): Prepends a doctype tag to the string output,
            i.e. something like ``<?xml version='1.0' encoding='UTF-8'?>``
    Returns:
        bytes: Bytestring containing raw XML representation of VOEvent.

    """
    vcopy = copy.deepcopy(voevent)
    _return_to_standard_xml(vcopy)
    s = etree.tostring(vcopy, pretty_print=pretty_print,
                       xml_declaration=xml_declaration,
                       encoding=encoding)
    return s