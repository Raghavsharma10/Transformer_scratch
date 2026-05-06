def _init_metadata(self):
        """stub"""
        self._confused_learning_objectives_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'confusedLearningObjectiveIds'),
            'element_label': 'Confused Learning Objectives',
            'instructions': 'List of IDs',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_list_values': [[]],
            'syntax': 'LIST'
        }
        self._feedbacks_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'feedbacks'),
            'element_label': 'Feedbacks',
            'instructions': 'Enter as many text feedback strings as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
            'object_set': []
        }