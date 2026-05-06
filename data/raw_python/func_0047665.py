def _init_metadata(self):
        """stub"""
        self._published_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'published'),
            'element_label': 'Published',
            'instructions': 'flags if item is published or not',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_published_values': [False],
            'syntax': 'BOOLEAN',
        }