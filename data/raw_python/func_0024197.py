def compare_field_caches(self, replica, original):
        """Verify original is subset of replica"""
        if original is None:
            original = []
        if replica is None:
            replica = []
        self.pr_dbg("Comparing orig with %s fields to replica with %s fields" %
                    (len(original), len(replica)))
        # convert list into dict, with each item's ['name'] as key
        orig = self.list_to_compare_dict(original)
        if orig is None:
            self.pr_dbg("Original has duplicate fields")
            return 1
        repl = self.list_to_compare_dict(replica)
        if repl is None:
            self.pr_dbg("Replica has duplicate fields")
            return 1
        # search orig for each item in repl
        # if any items in repl not within orig or vice versa, then complain
        # make sure contents of each item match
        orig_found = {}
        for (key, field) in iteritems(repl):
            field_name = field['name']
            if field_name not in orig:
                self.pr_dbg("Replica has field not found in orig %s: %s" %
                            (field_name, field))
                return 1
            orig_found[field_name] = True
            if orig[field_name] != field:
                self.pr_dbg("Field in replica doesn't match orig:")
                self.pr_dbg("orig:%s\nrepl:%s" % (orig[field_name], field))
                return 1
        unfound = set(orig_found.keys()) - set(repl.keys())
        if len(unfound) > 0:
            self.pr_dbg("Orig contains fields that were not in replica")
            self.pr_dbg('%s' % unfound)
            return 1
        # We don't care about case when replica has more fields than orig
        # unfound = set(repl.keys()) - set(orig_found.keys())
        # if len(unfound) > 0:
        #     self.pr_dbg("Replica contains fields that were not in orig")
        #     self.pr_dbg('%s' % unfound)
        #     return 1
        self.pr_dbg("Original matches replica")
        return 0