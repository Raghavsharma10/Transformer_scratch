def _init_metadata(self):
        """stub"""
        self._first_angle_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'first_angle'),
            'element_label': 'First Angle',
            'instructions': 'set boolean, is this a first angle projection',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_boolean_values': [False],
            'syntax': 'BOOLEAN',
        }