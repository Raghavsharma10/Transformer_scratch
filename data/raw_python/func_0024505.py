def put_object(self, obj):
        # TODO consider putting into a ES class
        self.pr_dbg('put_obj: %s' % self.json_dumps(obj))
        """
        Wrapper for es.index, determines metadata needed to index from obj.
        If you have a raw object json string you can hard code these:
        index is .kibana (as of kibana4);
        id can be A-Za-z0-9\- and must be unique;
        doc_type is either visualization, dashboard, search
            or for settings docs: config, or index-pattern.
        """
        if obj['_index'] is None or obj['_index'] == "":
            raise Exception("Invalid Object, no index")
        if obj['_id'] is None or obj['_id'] == "":
            raise Exception("Invalid Object, no _id")
        if obj['_type'] is None or obj['_type'] == "":
            raise Exception("Invalid Object, no _type")
        if obj['_source'] is None or obj['_source'] == "":
            raise Exception("Invalid Object, no _source")
        self.connect_es()
        self.es.indices.create(index=obj['_index'], ignore=400, timeout="2m")
        try:
            resp = self.es.index(index=obj['_index'],
                                 id=obj['_id'],
                                 doc_type=obj['_type'],
                                 body=obj['_source'], timeout="2m")
        except RequestError as e:
            self.pr_err('RequestError: %s, info: %s' % (e.error, e.info))
            raise
        return resp