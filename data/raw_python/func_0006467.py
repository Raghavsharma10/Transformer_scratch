def update(self, document_id, update_spec, namespace, timestamp):
        """Apply updates given in update_spec to the document whose id
        matches that of doc.

        """
        # Commit outstanding changes so that the document to be updated is the
        # same version to which the changes apply.
        self.commit()
        # Need to escape special characters in the document_id.
        document_id = ''.join(map(
            lambda c: '\\' + c if c in ESCAPE_CHARACTERS else c,
            u(document_id)
        ))

        query = "%s:%s" % (self.unique_key, document_id)
        results = self.solr.search(query)
        if not len(results):
            # Document may not be retrievable yet
            self.commit()
            results = self.solr.search(query)
        # Results is an iterable containing only 1 result
        for doc in results:
            # Remove metadata previously stored by Mongo Connector.
            doc.pop('ns')
            doc.pop('_ts')
            updated = self.apply_update(doc, update_spec)
            # A _version_ of 0 will always apply the update
            updated['_version_'] = 0
            self.upsert(updated, namespace, timestamp)
            return updated