def _init_metadata(self):
        """stub"""
        self._start_timestamp_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'start_timestamp'),
            'element_label': 'start timestamp',
            'instructions': 'enter an integer number of seconds for the start time',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'syntax': 'INTEGER',
            'minimum_integer': 0,
            'maximum_integer': None,
            'integer_set': [],
            'default_integer_values': [0]
        }
        self._end_timestamp_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'end_timestamp'),
            'element_label': 'end timestamp',
            'instructions': 'enter an integer number of seconds for the end time',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'syntax': 'INTEGER',
            'minimum_integer': 0,
            'maximum_integer': None,
            'integer_set': [],
            'default_integer_values': [0]
        }