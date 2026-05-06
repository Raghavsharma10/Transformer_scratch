def add_why(voevent, importance=None, expires=None, inferences=None):
    """Add Inferences, or set importance / expires attributes of the Why section.

    .. note::

        ``importance`` / ``expires`` are 'Why' attributes, therefore setting them
        will overwrite previous values.
        ``inferences``, on the other hand,  are appended to the list.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
        importance(float): Value from 0.0 to 1.0
        expires(datetime.datetime): Expiration date given inferred reason
            (See voevent spec).
        inferences(:class:`voeventparse.misc.Inference`): Inference or list of
            inferences, denoting probable identifications or associations, etc.
    """
    if not voevent.xpath('Why'):
        etree.SubElement(voevent, 'Why')
    if importance is not None:
        voevent.Why.attrib['importance'] = str(importance)
    if expires is not None:
        voevent.Why.attrib['expires'] = expires.replace(
            microsecond=0).isoformat()
    if inferences is not None:
        voevent.Why.extend(_listify(inferences))