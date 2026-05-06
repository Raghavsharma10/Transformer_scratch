def verb(**kwargs):
    """Create OAI-PMH envelope for response with verb."""
    e_tree, e_oaipmh = envelope(**kwargs)
    e_element = SubElement(e_oaipmh, etree.QName(NS_OAIPMH, kwargs['verb']))
    return e_tree, e_element