def run(self):
        """ Index the document. Since ids are predictable,
            we won't index anything twice. """
        with self.input().open() as handle:
            body = json.loads(handle.read())
        es = elasticsearch.Elasticsearch()
        id = body.get('_id')
        es.index(index='frontpage', doc_type='html', id=id, body=body)