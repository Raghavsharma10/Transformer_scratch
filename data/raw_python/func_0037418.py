def add_how(voevent, descriptions=None, references=None):
    """Add descriptions or references to the How section.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
        descriptions(str): Description string, or list of description
            strings.
        references(:py:class:`voeventparse.misc.Reference`): A reference element
            (or list thereof).
    """
    if not voevent.xpath('How'):
        etree.SubElement(voevent, 'How')
    if descriptions is not None:
        for desc in _listify(descriptions):
            # d = etree.SubElement(voevent.How, 'Description')
            # voevent.How.Description[voevent.How.index(d)] = desc
            ##Simpler:
            etree.SubElement(voevent.How, 'Description')
            voevent.How.Description[-1] = desc
    if references is not None:
        voevent.How.extend(_listify(references))