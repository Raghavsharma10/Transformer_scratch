def oaiid_fetcher(record_uuid, data):
    """Fetch a record's identifier.

    :param record_uuid: The record UUID.
    :param data: The record data.
    :returns: A :class:`invenio_pidstore.fetchers.FetchedPID` instance.
    """
    pid_value = data.get('_oai', {}).get('id')
    if pid_value is None:
        raise PersistentIdentifierError()
    return FetchedPID(
        provider=OAIIDProvider,
        pid_type=OAIIDProvider.pid_type,
        pid_value=str(pid_value),
    )