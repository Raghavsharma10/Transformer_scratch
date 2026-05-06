def transform_search_hit(self, pid, record_hit, links_factory=None):
        """Transform search result hit into an intermediate representation.

        :param pid: The :class:`invenio_pidstore.models.PersistentIdentifier`
            instance.
        :param record_hit: A dictionary containing a ``'_source'`` key with
            the record data.
        :param links_factory: The link factory. (Default: ``None``)
        :returns: The intermediate representation for the record.
        """
        return self.dump(self.preprocess_search_hit(pid, record_hit,
                         links_factory=links_factory))