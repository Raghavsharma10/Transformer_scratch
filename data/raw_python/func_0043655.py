def transform_record(self, pid, record, links_factory=None):
        """Transform record into an intermediate representation.

        :param pid: The :class:`invenio_pidstore.models.PersistentIdentifier`
            instance.
        :param record: The :class:`invenio_records.api.Record` instance.
        :param links_factory: The link factory. (Default: ``None``)
        :returns: The intermediate representation for the record.
        """
        return self.dump(self.preprocess_record(pid, record,
                         links_factory=links_factory))