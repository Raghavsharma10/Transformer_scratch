def listrecords(**kwargs):
    """Create OAI-PMH response for verb ListRecords."""
    record_dumper = serializer(kwargs['metadataPrefix'])

    e_tree, e_listrecords = verb(**kwargs)
    result = get_records(**kwargs)

    for record in result.items:
        pid = oaiid_fetcher(record['id'], record['json']['_source'])
        e_record = SubElement(e_listrecords,
                              etree.QName(NS_OAIPMH, 'record'))
        header(
            e_record,
            identifier=pid.pid_value,
            datestamp=record['updated'],
            sets=record['json']['_source'].get('_oai', {}).get('sets', []),
        )
        e_metadata = SubElement(e_record, etree.QName(NS_OAIPMH, 'metadata'))
        e_metadata.append(record_dumper(pid, record['json']))

    resumption_token(e_listrecords, result, **kwargs)
    return e_tree