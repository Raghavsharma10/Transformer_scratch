def dumps_etree(pid, record, **kwargs):
    """Dump MARC21 compatible record.

    :param pid: The :class:`invenio_pidstore.models.PersistentIdentifier`
        instance.
    :param record: The :class:`invenio_records.api.Record` instance.
    :returns: A LXML Element instance.
    """
    from dojson.contrib.to_marc21 import to_marc21
    from dojson.contrib.to_marc21.utils import dumps_etree

    return dumps_etree(to_marc21.do(record['_source']), **kwargs)