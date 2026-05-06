def add_record(self, record):
        """Add a record to the OAISet.

        :param record: Record to be added.
        :type record: `invenio_records.api.Record` or derivative.
        """
        record.setdefault('_oai', {}).setdefault('sets', [])

        assert not self.has_record(record)

        record['_oai']['sets'].append(self.spec)