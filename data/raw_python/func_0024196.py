def list_to_compare_dict(self, list_form):
        """Convert list into a data structure we can query easier"""
        compare_dict = {}
        for field in list_form:
            if field['name'] in compare_dict:
                self.pr_dbg("List has duplicate field %s:\n%s" %
                            (field['name'], compare_dict[field['name']]))
                if compare_dict[field['name']] != field:
                    self.pr_dbg("And values are different:\n%s" % field)
                return None
            compare_dict[field['name']] = field
            for ign_f in self.mappings_ignore:
                compare_dict[field['name']][ign_f] = 0
        return compare_dict