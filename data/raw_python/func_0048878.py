def _init_metadata(self):
        """Initialize metadata for this record"""
        self._enclosed_object_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'enclosed_object'),
            'element_label': 'Enclosed Object',
            'instructions': 'accepts an osid object Id',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }