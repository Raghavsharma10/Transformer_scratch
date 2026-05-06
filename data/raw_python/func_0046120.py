def _init_metadata(self):
        """stub"""
        self._min_decimal_value = None
        self._max_decimal_value = None
        self._color_coordinate_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'color_coordinate'),
            'element_label': 'Color Coordinate',
            'instructions': 'enter a color coordinate',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_coordinate_values': [{}],
            'syntax': 'COORDINATE',
            'coordinate_set': []
        }