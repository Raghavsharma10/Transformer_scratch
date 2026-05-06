def listsets(**kwargs):
    """Create OAI-PMH response for ListSets verb."""
    e_tree, e_listsets = verb(**kwargs)

    page = kwargs.get('resumptionToken', {}).get('page', 1)
    size = current_app.config['OAISERVER_PAGE_SIZE']
    oai_sets = OAISet.query.paginate(page=page, per_page=size, error_out=False)

    for oai_set in oai_sets.items:
        e_set = SubElement(e_listsets, etree.QName(NS_OAIPMH, 'set'))
        e_setSpec = SubElement(e_set, etree.QName(NS_OAIPMH, 'setSpec'))
        e_setSpec.text = oai_set.spec
        e_setName = SubElement(e_set, etree.QName(NS_OAIPMH, 'setName'))
        e_setName.text = sanitize_unicode(oai_set.name)
        if oai_set.description:
            e_setDescription = SubElement(e_set, etree.QName(NS_OAIPMH,
                                                             'setDescription'))
            e_dc = SubElement(
                e_setDescription, etree.QName(NS_OAIDC, 'dc'),
                nsmap=NSMAP_DESCRIPTION
            )
            e_dc.set(etree.QName(NS_XSI, 'schemaLocation'), NS_OAIDC)
            e_description = SubElement(e_dc, etree.QName(NS_DC, 'description'))
            e_description.text = oai_set.description

    resumption_token(e_listsets, oai_sets, **kwargs)
    return e_tree