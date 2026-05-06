def _init_metadata(self):
        """stub"""
        super(BaseMultiChoiceFileQuestionFormRecord, self)._init_metadata()
        self._choice_file_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choice-file'),
            'element_label': 'Choice File',
            'instructions': 'accepts an Asset Id',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }