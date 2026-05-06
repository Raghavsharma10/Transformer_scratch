def _init_map(self):
        """stub"""
        super(MultiLanguageMultipleChoiceQuestionFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['choices'] = \
            self._choices_metadata['default_object_values'][0]