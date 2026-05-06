def _init_metadata(self):
        """stub"""
        super(SimpleDifficultyItemFormRecord, self)._init_metadata()
        self._min_string_length = None
        self._max_string_length = None
        self._difficulty_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'text'),
            'element_label': 'Text',
            'instructions': 'enter a text string',
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