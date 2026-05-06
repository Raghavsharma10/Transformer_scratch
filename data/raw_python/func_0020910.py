def header(parent, identifier, datestamp, sets=None, deleted=False):
    """Attach ``<header/>`` element to a parent."""
    e_header = SubElement(parent, etree.QName(NS_OAIPMH, 'header'))
    if deleted:
        e_header.set('status', 'deleted')
    e_identifier = SubElement(e_header, etree.QName(NS_OAIPMH, 'identifier'))
    e_identifier.text = identifier
    e_datestamp = SubElement(e_header, etree.QName(NS_OAIPMH, 'datestamp'))
    e_datestamp.text = datetime_to_datestamp(datestamp)
    for spec in sets or []:
        e = SubElement(e_header, etree.QName(NS_OAIPMH, 'setSpec'))
        e.text = spec
    return e_header