def index_missing_documents(self, documents, request=None):
        """ Index documents that are missing from ES index.

        Determines which documents are missing using ES `mget` call which
        returns a list of document IDs as `documents`. Then missing
        `documents` from that list are indexed.
        """
        log.info('Trying to index documents of type `{}` missing from '
                 '`{}` index'.format(self.doc_type, self.index_name))
        if not documents:
            log.info('No documents to index')
            return
        query_kwargs = dict(
            index=self.index_name,
            doc_type=self.doc_type,
            fields=['_id'],
            body={'ids': [d['_pk'] for d in documents]},
        )
        try:
            response = self.api.mget(**query_kwargs)
        except IndexNotFoundException:
            indexed_ids = set()
        else:
            indexed_ids = set(
                d['_id'] for d in response['docs'] if d.get('found'))
        documents = [d for d in documents if str(d['_pk']) not in indexed_ids]

        if not documents:
            log.info('No documents of type `{}` are missing from '
                     'index `{}`'.format(self.doc_type, self.index_name))
            return

        self._bulk('index', documents, request)