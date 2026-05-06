def _init_metadata(self):
        """stub"""
        self._zone_conditions_metadata = {
            'zone_matches': Id(self.my_osid_object_form._authority,
                               self.my_osid_object_form._namespace,
                               'zone_conditions'),
            'element_label': 'zone conditions',
            'instructions': 'zone conditions for answer',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
        }
        self._coordinate_conditions_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'coordinate_conditions'),
            'element_label': 'coordinate conditions',
            'instructions': 'coordinate conditions for answer',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
        }
        self._spatial_unit_conditions_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'spatial_unit_conditions'),
            'element_label': 'spatial unit conditions',
            'instructions': 'spatial unit conditions for answer',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
        }