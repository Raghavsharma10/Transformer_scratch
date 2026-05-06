def add_citations(voevent, event_ivorns):
    """Add citations to other voevents.

    The schema mandates that the 'Citations' section must either be entirely
    absent, or non-empty - hence we require this wrapper function for its
    creation prior to listing the first citation.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
        event_ivorns (:class:`voeventparse.misc.EventIvorn`): List of EventIvorn
            elements to add to citation list.

    """
    if not voevent.xpath('Citations'):
        etree.SubElement(voevent, 'Citations')
    voevent.Citations.extend(_listify(event_ivorns))