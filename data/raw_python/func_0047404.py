def _init_metadata(self):
        """stub"""
        self._min_string_length = None
        self._max_string_length = None
        self._confused_learning_objectives_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'confusedLearningObjectiveIds'),
            'element_label': 'Confused Learning Objectives',
            'instructions': 'List of IDs',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_list_values': [[]],
            'syntax': 'LIST'
        }
        self._feedback_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'feedback'),
            'element_label': 'Feedback',
            'instructions': 'enter a feedback string',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_string_values': [{
                'text': '',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            }],
            'syntax': 'STRING',
            'minimum_string_length': self._min_string_length,
            'maximum_string_length': self._max_string_length,
            'string_set': []
        }