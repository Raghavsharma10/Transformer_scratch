def get_field_cache(self, cache_type='es'):
        """Return a list of fields' mappings"""
        if cache_type == 'kibana':
            try:
                search_results = urlopen(self.get_url).read().decode('utf-8')
            except HTTPError:  # as e:
                # self.pr_err("get_field_cache(kibana), HTTPError: %s" % e)
                return []
            index_pattern = json.loads(search_results)
            # Results look like: {"_index":".kibana","_type":"index-pattern","_id":"aaa*","_version":6,"found":true,"_source":{"title":"aaa*","fields":"<what we want>"}}  # noqa
            fields_str = index_pattern['_source']['fields']
            return json.loads(fields_str)
        elif cache_type == 'es' or cache_type.startswith('elastic'):
            search_results = urlopen(self.es_get_url).read().decode('utf-8')
            es_mappings = json.loads(search_results)
            # Results look like: {"<index_name>":{"mappings":{"<doc_type>":{"<field_name>":{"full_name":"<field_name>","mapping":{"<sub-field_name>":{"type":"date","index_name":"<sub-field_name>","boost":1.0,"index":"not_analyzed","store":false,"doc_values":false,"term_vector":"no","norms":{"enabled":false},"index_options":"docs","index_analyzer":"_date/16","search_analyzer":"_date/max","postings_format":"default","doc_values_format":"default","similarity":"default","fielddata":{},"ignore_malformed":false,"coerce":true,"precision_step":16,"format":"dateOptionalTime","null_value":null,"include_in_all":false,"numeric_resolution":"milliseconds","locale":""}}},  # noqa
            # now convert the mappings into the .kibana format
            field_cache = []
            for (index_name, val) in iteritems(es_mappings):
                if index_name != self.index:  # only get non-'.kibana' indices
                    # self.pr_dbg("index: %s" % index_name)
                    m_dict = es_mappings[index_name]['mappings']
                    # self.pr_dbg('m_dict %s' % m_dict)
                    mappings = self.get_index_mappings(m_dict)
                    # self.pr_dbg('mappings %s' % mappings)
                    field_cache.extend(mappings)
            field_cache = self.dedup_field_cache(field_cache)
            return field_cache
        self.pr_err("Unknown cache type: %s" % cache_type)
        return None