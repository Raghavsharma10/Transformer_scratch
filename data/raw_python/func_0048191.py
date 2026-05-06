def _init_map(self):
        """stub"""
        self.my_osid_object_form._my_map['learningObjectiveId'] = \
            str(self._learning_objective_id_metadata['default_id_values'][0])
        self.my_osid_object_form._my_map['minimumProficiency'] = \
            str(self._minimum_proficiency_metadata['default_id_values'][0])