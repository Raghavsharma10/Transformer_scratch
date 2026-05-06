def identify(**kwargs):
    """Create OAI-PMH response for verb Identify."""
    cfg = current_app.config

    e_tree, e_identify = verb(**kwargs)

    e_repositoryName = SubElement(
        e_identify, etree.QName(NS_OAIPMH, 'repositoryName'))
    e_repositoryName.text = cfg['OAISERVER_REPOSITORY_NAME']

    e_baseURL = SubElement(e_identify, etree.QName(NS_OAIPMH, 'baseURL'))
    e_baseURL.text = url_for('invenio_oaiserver.response', _external=True)

    e_protocolVersion = SubElement(e_identify,
                                   etree.QName(NS_OAIPMH, 'protocolVersion'))
    e_protocolVersion.text = cfg['OAISERVER_PROTOCOL_VERSION']

    for adminEmail in cfg['OAISERVER_ADMIN_EMAILS']:
        e = SubElement(e_identify, etree.QName(NS_OAIPMH, 'adminEmail'))
        e.text = adminEmail

    e_earliestDatestamp = SubElement(
        e_identify, etree.QName(
            NS_OAIPMH, 'earliestDatestamp'))
    earliest_date = datetime(MINYEAR, 1, 1)
    earliest_record = OAIServerSearch(
        index=current_app.config['OAISERVER_RECORD_INDEX']).sort({
            "_created": {"order": "asc"}})[0:1].execute()
    if len(earliest_record.hits.hits) > 0:
        created_date_str = earliest_record.hits.hits[0].get(
            "_source", {}).get('_created')
        if created_date_str:
            earliest_date = arrow.get(
                created_date_str).to('utc').datetime.replace(tzinfo=None)

    e_earliestDatestamp.text = datetime_to_datestamp(earliest_date)

    e_deletedRecord = SubElement(e_identify,
                                 etree.QName(NS_OAIPMH, 'deletedRecord'))
    e_deletedRecord.text = 'no'

    e_granularity = SubElement(e_identify,
                               etree.QName(NS_OAIPMH, 'granularity'))
    assert cfg['OAISERVER_GRANULARITY'] in DATETIME_FORMATS
    e_granularity.text = cfg['OAISERVER_GRANULARITY']

    compressions = cfg['OAISERVER_COMPRESSIONS']
    if compressions != ['identity']:
        for compression in compressions:
            e_compression = SubElement(e_identify,
                                       etree.QName(NS_OAIPMH, 'compression'))
            e_compression.text = compression

    for description in cfg.get('OAISERVER_DESCRIPTIONS', []):
        e_description = SubElement(e_identify,
                                   etree.QName(NS_OAIPMH, 'description'))
        e_description.append(etree.fromstring(description))

    return e_tree