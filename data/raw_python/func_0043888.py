def serialize(self, pid, record, links_factory=None):
        """Serialize a single record and persistent identifier.

        :param pid: The :class:`invenio_pidstore.models.PersistentIdentifier`
            instance.
        :param record: The :class:`invenio_records.api.Record` instance.
        :param links_factory: Factory function for the link generation,
            which are added to the response.
        :returns: The object serialized.
        """
        return dumps(self.transform_record(pid, record, links_factory),
                     **self.dumps_kwargs)