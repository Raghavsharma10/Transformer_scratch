def get_doc_type_mappings(self, doc_type):
        """Converts all doc_types' fields to .kibana"""
        doc_fields_arr = []
        found_score = False
        for (key, val) in iteritems(doc_type):
            # self.pr_dbg("\t\tfield: %s" % key)
            # self.pr_dbg("\tval: %s" % val)
            add_it = False
            retdict = {}
            # _ are system
            if not key.startswith('_'):
                if 'mapping' not in doc_type[key]:
                    self.pr_err("No mapping in doc_type[%s]" % key)
                    return None
                if key in doc_type[key]['mapping']:
                    subkey_name = key
                else:
                    subkey_name = re.sub('.*\.', '', key)
                if subkey_name not in doc_type[key]['mapping']:
                    self.pr_err(
                        "Couldn't find subkey " +
                        "doc_type[%s]['mapping'][%s]" % (key, subkey_name))
                    return None
                # self.pr_dbg("\t\tsubkey_name: %s" % subkey_name)
                retdict = self.get_field_mappings(
                    doc_type[key]['mapping'][subkey_name])
                add_it = True
            # system mappings don't list a type,
            # but kibana makes them all strings
            if key in self.sys_mappings:
                retdict['analyzed'] = False
                retdict['indexed'] = False
                if key == '_source':
                    retdict = self.get_field_mappings(
                        doc_type[key]['mapping'][key])
                    retdict['type'] = "_source"
                elif key == '_score':
                    retdict['type'] = "number"
                elif 'type' not in retdict:
                    retdict['type'] = "string"
                add_it = True
            if add_it:
                retdict['name'] = key
                retdict['count'] = 0  # always init to 0
                retdict['scripted'] = False  # I haven't observed a True yet
                if not self.check_mapping(retdict):
                    self.pr_err("Error, invalid mapping")
                    return None
                # the fields element is an escaped array of json
                # make the array here, after all collected, then escape it
                doc_fields_arr.append(retdict)
        if not found_score:
            doc_fields_arr.append(
                {"name": "_score",
                 "type": "number",
                 "count": 0,
                 "scripted": False,
                 "indexed": False,
                 "analyzed": False,
                 "doc_values": False})
        return doc_fields_arr