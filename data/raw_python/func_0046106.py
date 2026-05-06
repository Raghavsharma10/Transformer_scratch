def _init_metadata(self):
        """stub"""
        self._files_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'files'),
            'element_label': 'Files',
            'instructions': 'enter a file id with optional label',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._file_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'file'),
            'element_label': 'File',
            'instructions': 'accepts an Asset Id',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
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