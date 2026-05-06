def is_kibana_cache_incomplete(self, es_cache, k_cache):
        """Test if k_cache is incomplete

        Assume k_cache is always correct, but could be missing new
        fields that es_cache has
        """
        # convert list into dict, with each item's ['name'] as key
        k_dict = {}
        for field in k_cache:
            # self.pr_dbg("field: %s" % field)
            k_dict[field['name']] = field
            for ign_f in self.mappings_ignore:
                k_dict[field['name']][ign_f] = 0
        es_dict = {}
        for field in es_cache:
            es_dict[field['name']] = field
            for ign_f in self.mappings_ignore:
                es_dict[field['name']][ign_f] = 0
        es_set = set(es_dict.keys())
        k_set = set(k_dict.keys())
        # reasons why kibana cache could be incomplete:
        #     k_dict is missing keys that are within es_dict
        #     We don't care if k has keys that es doesn't
        # es {1,2} k {1,2,3}; intersection {1,2}; len(es-{}) 0
        # es {1,2} k {1,2};   intersection {1,2}; len(es-{}) 0
        # es {1,2} k {};      intersection {};    len(es-{}) 2
        # es {1,2} k {1};     intersection {1};   len(es-{}) 1
        # es {2,3} k {1};     intersection {};    len(es-{}) 2
        # es {2,3} k {1,2};   intersection {2};   len(es-{}) 1
        return len(es_set - k_set.intersection(es_set)) > 0