def remove_record(self, record):
        """Remove a record from the OAISet.

        :param record: Record to be removed.
        :type record: `invenio_records.api.Record` or derivative.
        """
        assert self.has_record(record)

        record['_oai']['sets'] = [
            s for s in record['_oai']['sets'] if s != self.spec]