def serialize_oaipmh(self, pid, record):
        """Serialize a single record for OAI-PMH.

        :param pid: The :class:`invenio_pidstore.models.PersistentIdentifier`
            instance.
        :param record: The :class:`invenio_records.api.Record` instance.
        :returns: The object serialized.
        """
        obj = self.transform_record(pid, record['_source']) \
            if isinstance(record['_source'], Record) \
            else self.transform_search_hit(pid, record)

        return dumps_etree(obj, **self.dumps_kwargs)