def _init_metadata(self):
        """stub"""
        self._min_integer_value = None
        self._max_integer_value = None
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
            'default_integer_values': [None],
            'syntax': 'INTEGER',
            'minimum_integer': self._min_integer_value,
            'maximum_integer': self._max_integer_value,
            'integer_set': []
        }