def _init_metadata(self):
        """stub"""
        self._choices_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choices'),
            'element_label': 'Choices',
            'instructions': 'Enter as many choices as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [''],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._choice_name_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'question_string'),
            'element_label': 'choice name',
            'instructions': 'enter a short label for this choice',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_string_values': [''],
            'syntax': 'STRING',
            'minimum_string_length': 0,
            'maximum_string_length': 1024,
            'string_set': []
        }
        self._multi_answer_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'multi_answer'),
            'element_label': 'Is Multi-Answer',
            'instructions': 'accepts a boolean (True/False) value',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': ['False'],
            'syntax': 'BOOLEAN',
            'id_set': []
        }