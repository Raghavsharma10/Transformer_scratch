def field_cache_to_index_pattern(self, field_cache):
        """Return a .kibana index-pattern doc_type"""
        mapping_dict = {}
        mapping_dict['customFormats'] = "{}"
        mapping_dict['title'] = self.index_pattern
        # now post the data into .kibana
        mapping_dict['fields'] = json.dumps(field_cache, separators=(',', ':'))
        # in order to post, we need to create the post string
        mapping_str = json.dumps(mapping_dict, separators=(',', ':'))
        return mapping_str