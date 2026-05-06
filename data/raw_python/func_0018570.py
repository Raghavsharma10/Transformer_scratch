def remove(self, document_id, namespace, timestamp):
        """Remove a document from Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)

        action = {
            '_op_type': 'delete',
            '_index': index,
            '_type': doc_type,
            '_id': u(document_id)
        }

        meta_action = {
            '_op_type': 'delete',
            '_index': self.meta_index_name,
            '_type': self.meta_type,
            '_id': u(document_id)
        }

        self.index(action, meta_action)