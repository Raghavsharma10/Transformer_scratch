def getrecord(**kwargs):
    """Create OAI-PMH response for verb Identify."""
    record_dumper = serializer(kwargs['metadataPrefix'])
    pid = OAIIDProvider.get(pid_value=kwargs['identifier']).pid
    record = Record.get_record(pid.object_uuid)

    e_tree, e_getrecord = verb(**kwargs)
    e_record = SubElement(e_getrecord, etree.QName(NS_OAIPMH, 'record'))

    header(
        e_record,
        identifier=pid.pid_value,
        datestamp=record.updated,
        sets=record.get('_oai', {}).get('sets', []),
    )
    e_metadata = SubElement(e_record,
                            etree.QName(NS_OAIPMH, 'metadata'))
    e_metadata.append(record_dumper(pid, {'_source': record}))

    return e_tree