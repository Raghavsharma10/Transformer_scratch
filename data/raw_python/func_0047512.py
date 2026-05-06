def _init_map(self):
        """stub"""
        super(TextAnswerFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['minStringLength'] = \
            self._min_string_length_metadata['default_cardinal_values'][0]
        self.my_osid_object_form._my_map['maxStringLength'] = \
            self._max_string_length_metadata['default_cardinal_values'][0]