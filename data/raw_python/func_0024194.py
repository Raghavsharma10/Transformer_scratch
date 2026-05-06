def get_field_mappings(self, field):
        """Converts ES field mappings to .kibana field mappings"""
        retdict = {}
        retdict['indexed'] = False
        retdict['analyzed'] = False
        for (key, val) in iteritems(field):
            if key in self.mappings:
                if (key == 'type' and
                    (val == "long" or
                     val == "integer" or
                     val == "double" or
                     val == "float")):
                    val = "number"
                # self.pr_dbg("\t\t\tkey: %s" % key)
                # self.pr_dbg("\t\t\t\tval: %s" % val)
                retdict[key] = val
            if key == 'index' and val != "no":
                retdict['indexed'] = True
                # self.pr_dbg("\t\t\tkey: %s" % key)
                # self.pr_dbg("\t\t\t\tval: %s" % val)
                if val == "analyzed":
                    retdict['analyzed'] = True
        return retdict