def set_author(voevent, title=None, shortName=None, logoURL=None,
               contactName=None, contactEmail=None, contactPhone=None,
               contributor=None):
    """For setting fields in the detailed author description.

    This can optionally be neglected if a well defined AuthorIVORN is supplied.

    .. note:: Unusually for this library,
        the args here use CamelCase naming convention,
        since there's a direct mapping to the ``Author.*``
        attributes to which they will be assigned.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
            The rest of the arguments are strings corresponding to child elements.
    """
    # We inspect all local variables except the voevent packet,
    # Cycling through and assigning them on the Who.Author element.
    AuthChildren = locals()
    AuthChildren.pop('voevent')
    if not voevent.xpath('Who/Author'):
        etree.SubElement(voevent.Who, 'Author')
    for k, v in AuthChildren.items():
        if v is not None:
            voevent.Who.Author[k] = v