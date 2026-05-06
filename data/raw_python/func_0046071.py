def _init_metadata(self):
        """stub"""
        self._min_string_length = None
        self._max_string_length = None
        self._texts_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'texts'),
            'element_label': 'Texts',
            'instructions': 'enter text with optional label',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._text_metadata = {
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