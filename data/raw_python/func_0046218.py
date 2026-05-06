def _init_metadata(self):
        """stub"""
        self._review_options_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'review_options'),
            'element_label': 'Review Options',
            'instructions': 'Choose various Review Options',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._whether_correct_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'report_correct'),
            'element_label': 'Report Correct',
            'instructions': 'Choose when to report correct answer to Taker',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._solutions_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'review_solutions'),
            'element_label': 'Review Solutions / Explanations',
            'instructions': 'Choose when to report a solution or explanation text blob, when available',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._during_attempt_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'during-attempt'),
            'element_label': 'During Attempt',
            'instructions': 'accepts a boolean (True/False) value',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._after_attempt_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'during-attempt'),
            'element_label': 'During Attempt',
            'instructions': 'accepts a boolean (True/False) value',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._before_deadline_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'during-attempt'),
            'element_label': 'During Attempt',
            'instructions': 'accepts a boolean (True/False) value',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._after_deadline_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'during-attempt'),
            'element_label': 'During Attempt',
            'instructions': 'accepts a boolean (True/False) value',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._min_max_attempts_value = None
        self._max_max_attempts_value = None
        self._max_attempts_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'max_attempts'),
            'element_label': 'Maximum Attempts',
            'instructions': 'enter an integer value for maximum attempts',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_integer_values': [None],
            'syntax': 'INTEGER',
            'minimum_integer': self._min_max_attempts_value,
            'maximum_integer': self._max_max_attempts_value,
            'integer_set': []
        }