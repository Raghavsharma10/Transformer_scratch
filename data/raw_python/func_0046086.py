def _init_metadata(self):
        """stub"""
        self._attempts_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'attempts'),
            'element_label': 'Attempts',
            'instructions': 'Max number of student attempts',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [0],
            'syntax': 'INTEGER',
            'object_set': [],
            'minimum_integer': None,
            'maximum_integer': None,
            'integer_set': []
        }
        self._weight_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'weight'),
            'element_label': 'Weight',
            'instructions': 'Weight of the item when calculating grades',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [1.0],
            'syntax': 'DECIMAL',
            'object_set': [],
            'decimal_scale': None,
            'minimum_decimal': None,
            'maximum_decimal': None,
            'decimal_set': []
        }
        # self._rerandomize_metadata = {
        #     'element_id': Id(self.my_osid_object_form._authority,
        #                      self.my_osid_object_form._namespace,
        #                      'rerandomize'),
        #     'element_label': 'Randomize',
        #     'instructions': 'How to rerandomize the parameters',
        #     'required': False,
        #     'read_only': False,
        #     'linked': False,
        #     'array': False,
        #     'default_object_values': ['never'],
        #     'syntax': 'STRING',
        #     'minimum_string_length': None,
        #     'maximum_string_length': None,
        #     'string_set': []
        # }
        self._showanswer_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'showanswer'),
            'element_label': 'Show answer',
            'instructions': 'When to show the answer to the student',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': ['closed'],
            'syntax': 'STRING',
            'minimum_string_length': None,
            'maximum_string_length': None,
            'string_set': []
        }
        self._markdown_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'markdown'),
            'element_label': 'Studio markdown',
            'instructions': 'Studio markdown representation of the problem',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [''],
            'syntax': 'STRING',
            'minimum_string_length': None,
            'maximum_string_length': None,
            'string_set': []
        }