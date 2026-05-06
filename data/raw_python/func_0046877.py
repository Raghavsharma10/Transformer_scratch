def _init_metadata(self):
        """stub"""
        self._n_of_m_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'nOfM'),
            'element_label': 'nOfM',
            'instructions': 'Student is expected to do N of M questions',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [-1],
            'syntax': 'INTEGER',
            'object_set': [],
            'minimum_integer': None,
            'maximum_integer': None,
            'integer_set': []
        }