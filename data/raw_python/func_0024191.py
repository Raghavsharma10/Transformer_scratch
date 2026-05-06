def check_mapping(self, m):
        """Assert minimum set of fields in cache, does not validate contents"""
        if 'name' not in m:
            self.pr_dbg("Missing %s" % "name")
            return False
        # self.pr_dbg("Checking %s" % m['name'])
        for x in ['analyzed', 'indexed', 'type', 'scripted', 'count']:
            if x not in m or m[x] == "":
                self.pr_dbg("Missing %s" % x)
                self.pr_dbg("Full %s" % m)
                return False
        if 'doc_values' not in m or m['doc_values'] == "":
            if not m['name'].startswith('_'):
                self.pr_dbg("Missing %s" % "doc_values")
                return False
            m['doc_values'] = False
        return True