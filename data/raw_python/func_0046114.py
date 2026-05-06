def _init_metadata(self):
        """stub"""
        self._file_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'file'),
            'element_label': 'File',
            'instructions': 'accepts an asset id and optional asset_content type',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }