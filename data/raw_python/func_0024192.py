def get_index_mappings(self, index):
        """Converts all index's doc_types to .kibana"""
        fields_arr = []
        for (key, val) in iteritems(index):
            # self.pr_dbg("\tdoc_type: %s" % key)
            doc_mapping = self.get_doc_type_mappings(index[key])
            # self.pr_dbg("\tdoc_mapping: %s" % doc_mapping)
            if doc_mapping is None:
                return None
            # keep adding to the fields array
            fields_arr.extend(doc_mapping)
        return fields_arr