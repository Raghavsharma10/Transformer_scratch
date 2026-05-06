def get_objects(self, search_field, search_val):
        """Return all objects of type (assumes < MAX_HITS)"""
        query = ("{ size: " + str(self.max_hits) + ", " +
                 "query: { filtered: { filter: { " +
                 search_field + ": { value: \"" + search_val + "\"" +
                 " } } } } } }")
        self.connect_es()
        res = self.es.search(index=self.index, body=query)
        # self.pr_dbg("%d Hits:" % res['hits']['total'])
        objects = {}
        for doc in res['hits']['hits']:
            objects[doc['_id']] = {}
            # To make uploading easier in the future:
            # Record all those bits into the backup.
            # Mimics how ES returns the result.
            # Prevents having to store this in some external, contrived, format
            objects[doc['_id']]['_index'] = self.index  # also in doc['_index']
            objects[doc['_id']]['_type'] = doc['_type']
            objects[doc['_id']]['_id'] = doc['_id']
            objects[doc['_id']]['_source'] = doc['_source']  # the actual result
        return objects