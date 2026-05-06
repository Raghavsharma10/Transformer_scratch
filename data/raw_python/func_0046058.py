def _init_metadata(self):
        """stub"""
        self._provenance_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'provenanceId'),
            'element_label': 'provenanceId',
            'instructions': 'The item that "gave birth" to this item.',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [''],
            'syntax': 'STRING',
            'minimum_string_length': None,
            'maximum_string_length': None,
            'string_set': []
        }