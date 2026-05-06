def _init_metadata(self):
        """stub"""
        self._min_integer_value = None
        self._max_integer_value = None
        self._integer_values_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'integer_values'),
            'element_label': 'Integer Values',
            'instructions': 'enter integer values with optional labels',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._integer_value_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'integer_value'),
            'element_label': 'Integer Value',
            'instructions': 'enter an integer value',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_integer_value': [None],
            'syntax': 'INTEGER',
            'minimum_integer': self._min_integer_value,
            'maximum_integer': self._max_integer_value,
            'integer_set': []
        }
        self._label_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'label'),
            'element_label': 'Label',
            'instructions': 'enter a string',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_string_values': [str(ObjectId())],
            'syntax': 'STRING',
            'minimum_string_length': 0,
            'maximum_string_length': 128,
            'string_set': []
        }