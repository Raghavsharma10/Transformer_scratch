def _init_metadata(self):
        """stub"""
        super(TextAnswerFormRecord, self)._init_metadata()
        self._min_string_length_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'min-string-length'),
            'element_label': 'min string length',
            'instructions': 'enter minimum string length',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_cardinal_values': [self._min_string_length],
            'syntax': 'CARDINAL',
            'minimum_cardinal': None,
            'maximum_cardinal': None,
            'cardinal_set': []
        }
        self._max_string_length_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'max-string-length'),
            'element_label': 'max string length',
            'instructions': 'enter maximum string length',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_cardinal_values': [self._max_string_length],
            'syntax': 'CARDINAL',
            'minimum_cardinal': None,
            'maximum_cardinal': None,
            'cardinal_set': []
        }