def _init_map(self):
        """stub"""
        super(EdXCompositionFormRecord, self)._init_map()
        TextsFormRecord._init_map(self)  # because the OsidForm breaks the MRO chain for super, in TemporalFormRecord
        ProvenanceFormRecord._init_map(self)  # because the OsidForm breaks the MRO chain for super, in TemporalFormRecord

        self.my_osid_object_form._my_map['texts']['fileName'] = \
            self._text_metadata['default_string_values'][0]
        self.my_osid_object_form._my_map['texts']['format'] = \
            self._text_metadata['default_string_values'][0]  # homework, exam, lab, etc.
        self.my_osid_object_form._my_map['visibleToStudents'] = \
            self._visible_to_students_metadata['default_boolean_values'][0]
        self.my_osid_object_form._my_map['draft'] = \
            self._draft_metadata['default_boolean_values'][0]
        self.my_osid_object_form._my_map['texts']['userPartitionId'] = \
            self._text_metadata['default_string_values'][0]
        self.my_osid_object_form._my_map['texts']['org'] = \
            self._text_metadata['default_string_values'][0]
        self.my_osid_object_form._my_map['learningObjectiveIds'] = \
            self._learning_objective_ids_metadata['default_string_values'][0]