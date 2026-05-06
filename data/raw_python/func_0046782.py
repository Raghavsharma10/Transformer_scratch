def _init_map(self):
        """stub"""
        # super(ScaffoldDownAssessmentPartFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['itemIds'] = \
            [str(self._item_ids_metadata['default_id_values'][0])]
        self.my_osid_object_form._my_map['learningObjectiveIds'] = \
            [str(self._learning_objective_ids_metadata['default_id_values'][0])]
        self.my_osid_object_form._my_map['maxLevels'] = \
            self._max_levels_metadata['default_cardinal_values'][0]
        self.my_osid_object_form._my_map['maxWaypointItems'] = \
            self._max_waypoint_items_metadata['default_cardinal_values'][0]
        self.my_osid_object_form._my_map['waypointQuota'] = \
            self._waypoint_quota_metadata['default_cardinal_values'][0]
        self.my_osid_object_form._my_map['itemBankId'] = \
            self._item_bank_id_metadata['default_id_values'][0]
        self.my_osid_object_form._my_map['allowRepeatItems'] = \
            bool(self._allow_repeat_items_metadata['default_boolean_values'][0])