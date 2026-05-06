def _init_metadata(self):
        """stub"""
        self._learning_objective_id_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'learning_objective_id'),
            'element_label': 'Learning Objective Id',
            'instructions': 'accepts a valid OSID Id string',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }
        self._minimum_proficiency_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'minimum_proficiency'),
            'element_label': 'Minimum Proficiency in the given Objective to "pass"',
            'instructions': 'accepts a valid OSID Id string',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }