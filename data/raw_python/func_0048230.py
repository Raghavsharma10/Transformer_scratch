def _init_metadata(self):
        """stub"""
        self._choice_ids_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choice_ids'),
            'element_label': 'response set',
            'instructions': 'submit correct choice for answer',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
        }
        self._choice_id_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'choice_id'),
            'element_label': 'response set',
            'instructions': 'submit correct choice for answer',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }