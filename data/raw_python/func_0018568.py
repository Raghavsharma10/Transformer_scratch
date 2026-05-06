def upsert(self, doc, namespace, timestamp, update_spec=None):
        """Insert a document into Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)
        # No need to duplicate '_id' in source document
        doc_id = u(doc.pop("_id"))
        metadata = {
            'ns': namespace,
            '_ts': timestamp
        }

        # Index the source document, using lowercase namespace as index name.
        action = {
            '_op_type': 'index',
            '_index': index,
            '_type': doc_type,
            '_id': doc_id,
            '_source': self._formatter.format_document(doc)
        }
        # Index document metadata with original namespace (mixed upper/lower).
        meta_action = {
            '_op_type': 'index',
            '_index': self.meta_index_name,
            '_type': self.meta_type,
            '_id': doc_id,
            '_source': bson.json_util.dumps(metadata)
        }

        self.index(action, meta_action, doc, update_spec)

        # Leave _id, since it's part of the original document
        doc['_id'] = doc_id