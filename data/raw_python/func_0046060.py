def _init_metadata(self):
        """stub"""
        self._resource_id_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'resource_id'),
            'element_label': 'Resource Id',
            'instructions': 'accepts a valid OSID Id',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }