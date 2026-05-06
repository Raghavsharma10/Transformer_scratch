def _init_metadata(self):
        """stub"""
        super(EdXCompositionFormRecord, self)._init_metadata()
        TextsFormRecord._init_metadata(self)  # because the OsidForm breaks the MRO chain for super, in TemporalFormRecord
        ProvenanceFormRecord._init_metadata(self)  # because the OsidForm breaks the MRO chain for super, in TemporalFormRecord
        self._visible_to_students_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'visible_to_students'),
            'element_label': 'Visible to students',
            'instructions': 'enter a boolean value',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_boolean_values': [True],
            'syntax': 'BOOLEAN'
        }
        self._draft_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'draft'),
            'element_label': 'Draft',
            'instructions': 'enter a boolean value',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_boolean_values': [False],
            'syntax': 'BOOLEAN'
        }

        # ideally this would be type LIST?
        self._learning_objective_ids_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'learning_objectives'),
            'element_label': 'learning_objectives',
            'instructions': 'enter a list of strings',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': True,
            'default_string_values': [[]],
            'syntax': 'STRING',
            'minimum_string_length': self._min_string_length,
            'maximum_string_length': self._max_string_length,
            'string_set': []
        }