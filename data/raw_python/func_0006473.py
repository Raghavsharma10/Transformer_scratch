def get_last_doc(self):
        """Returns the last document stored in the Solr engine.
        """
        #search everything, sort by descending timestamp, return 1 row
        try:
            result = self.solr.search('*:*', sort='_ts desc', rows=1)
        except ValueError:
            return None

        for r in result:
            r['_id'] = r.pop(self.unique_key)
            return r