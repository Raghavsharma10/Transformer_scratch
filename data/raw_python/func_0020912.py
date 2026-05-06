def listidentifiers(**kwargs):
    """Create OAI-PMH response for verb ListIdentifiers."""
    e_tree, e_listidentifiers = verb(**kwargs)
    result = get_records(**kwargs)

    for record in result.items:
        pid = oaiid_fetcher(record['id'], record['json']['_source'])
        header(
            e_listidentifiers,
            identifier=pid.pid_value,
            datestamp=record['updated'],
            sets=record['json']['_source'].get('_oai', {}).get('sets', []),
        )

    resumption_token(e_listidentifiers, result, **kwargs)
    return e_tree