def _init_metadata(self):
        """stub"""
        self._min_decimal_value = None
        self._max_decimal_value = None
        self._decimal_values_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'decimal_values'),
            'element_label': 'Decimal Values',
            'instructions': 'enter decimal values with optional labels',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._decimal_value_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'decimal_value'),
            'element_label': 'Decimal Value',
            'instructions': 'enter a decimal value',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_decimal_values': [None, 0.0],
            'syntax': 'DECIMAL',
            'decimal_scale': None,
            'minimum_decimal': self._min_decimal_value,
            'maximum_decimal': self._max_decimal_value,
            'decimal_set': []
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