def _to_solr(self, data):
        '''
        Sends data to a Solr instance.
        '''
        return self._dest.index_json(self._dest_coll, json.dumps(data,sort_keys=True))