def _remove_root_tag_prefix(v):
    """
    Removes 'voe' namespace prefix from root tag.

    When we load in a VOEvent, the root element has a tag prefixed by
     the VOE namespace, e.g. {http://www.ivoa.net/xml/VOEvent/v2.0}VOEvent
    Because objectify expects child elements to have the same namespace as
    their parent, this breaks the python-attribute style access mechanism.
    We can get around it without altering root, via e.g
     who = v['{}Who']

    Alternatively, we can temporarily ditch the namespace altogether.
    This makes access to elements easier, but requires care to reinsert
    the namespace upon output.

    I've gone for the latter option.
    """
    if v.prefix:
        # Create subelement without a prefix via etree.SubElement
        etree.SubElement(v, 'original_prefix')
        # Now carefully access said named subelement (without prefix cascade)
        # and alter the first value in the list of children with this name...
        # LXML syntax is a minefield!
        v['{}original_prefix'][0] = v.prefix
        v.tag = v.tag.replace(''.join(('{', v.nsmap[v.prefix], '}')), '')
        # Now v.tag = '{}VOEvent', v.prefix = None
    return