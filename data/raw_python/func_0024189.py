def post_field_cache(self, field_cache):
        """Where field_cache is a list of fields' mappings"""
        index_pattern = self.field_cache_to_index_pattern(field_cache)
        # self.pr_dbg("request/post: %s" % index_pattern)
        resp = requests.post(self.post_url, data=index_pattern).text
        # resp = {"_index":".kibana","_type":"index-pattern","_id":"aaa*","_version":1,"created":true}  # noqa
        resp = json.loads(resp)
        return 0