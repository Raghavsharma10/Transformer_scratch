def _init_map(self):
        """stub"""
        super(SourceItemFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['texts']['source'] = \
            self._source_metadata['default_string_values'][0]