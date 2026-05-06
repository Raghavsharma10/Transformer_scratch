def _init_metadata(self):
        """stub"""
        super(BaseMultiChoiceTextQuestionFormRecord, self)._init_metadata()
        self._choice_text_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choice_text'),
            'element_label': 'choice text',
            'instructions': 'enter the text for this choice',
            'required': True,
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
            'minimum_string_length': 0,
            'maximum_string_length': 1024,
            'string_set': []
        }