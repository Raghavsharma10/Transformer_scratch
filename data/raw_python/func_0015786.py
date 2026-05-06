def raw(self):
        """
        Build query and passes to `Elasticsearch`, then returns the raw
        format returned.
        """
        es = self.get_es()

        params = dict(self.query_params)
        mlt_fields = self.mlt_fields or params.pop('mlt_fields', [])

        body = self.s.build_search() if self.s else ''

        hits = es.mlt(
            index=self.index, doc_type=self.doctype, id=self.id,
            mlt_fields=mlt_fields, body=body, **params)

        log.debug(hits)

        return hits