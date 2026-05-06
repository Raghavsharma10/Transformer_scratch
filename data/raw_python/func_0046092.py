def _init_metadata(self):
        """stub"""
        self._min_time_value = None
        self._max_time_value = None
        self._time_value_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'time_value'),
            'element_label': 'Time Value',
            'instructions': 'enter a time duration string / duration',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_duration_values': [{
                'hours': 0,
                'minutes': 0,
                'seconds': 0
            }],
            'syntax': 'DURATION',
            'minimum_time': self._min_time_value,
            'maximum_time': self._max_time_value
        }