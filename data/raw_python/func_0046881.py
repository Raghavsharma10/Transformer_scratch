def _init_metadata(self):
        """stub"""
        self._min_string_length = None
        self._max_string_length = None
        self._unlock_previous_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'unlock_previous'),
            'element_label': 'unlock_previous',
            'instructions': 'Indicator to UI on how to treat the previous button',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_string_values': ['always'],
            'syntax': 'STRING',
            'minimum_string_length': self._min_string_length,
            'maximum_string_length': self._max_string_length,
            'string_set': []
        }