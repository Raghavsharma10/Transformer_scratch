def update(self, document_id, update_spec, namespace, timestamp):
        """Apply updates given in update_spec to the document whose id
        matches that of doc.
        """

        index, doc_type = self._index_and_mapping(namespace)
        with self.lock:
            # Check if document source is stored in local buffer
            document = self.BulkBuffer.get_from_sources(index,
                                                        doc_type,
                                                        u(document_id))
        if document:
            # Document source collected from local buffer
            # Perform apply_update on it and then it will be
            # ready for commiting to Elasticsearch
            updated = self.apply_update(document, update_spec)
            # _id is immutable in MongoDB, so won't have changed in update
            updated['_id'] = document_id
            self.upsert(updated, namespace, timestamp)
        else:
            # Document source needs to be retrieved from Elasticsearch
            # before performing update. Pass update_spec to upsert function
            updated = {"_id": document_id}
            self.upsert(updated, namespace, timestamp, update_spec)
        # upsert() strips metadata, so only _id + fields in _source still here
        return updated