def _init_metadata(self):
        """stub"""
        super(MultiLanguageDragAndDropQuestionFormRecord, self)._init_metadata()
        self._droppables_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'droppables'),
            'element_label': 'droppables',
            'instructions': 'Enter as many droppables as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._targets_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'targets'),
            'element_label': 'targets',
            'instructions': 'Enter as many targets as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._zones_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'zones'),
            'element_label': 'zones',
            'instructions': 'Enter as many zones as you wish',
            'required': True,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_object_values': [[]],
            'syntax': 'OBJECT',
            'object_set': []
        }
        self._shuffle_droppables_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'shuffleDroppables'),
            'element_label': 'Shuffle Droppables',
            'instructions': 'Shuffle droppables',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._shuffle_targets_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'shuffleTargets'),
            'element_label': 'Shuffle Targets',
            'instructions': 'Shuffle targets',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }
        self._shuffle_zones_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'shuffleZones'),
            'element_label': 'Shuffle Zones',
            'instructions': 'Shuffle zones',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN',
        }