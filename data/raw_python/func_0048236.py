def _init_metadata(self):
        """stub"""
        super(MultiLanguageMultipleChoiceQuestionFormRecord, self)._init_metadata()
        self._choices_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choices'),
            'element_label': 'choices',
            'instructions': 'Enter as many text choices as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
            'object_set': []
        }