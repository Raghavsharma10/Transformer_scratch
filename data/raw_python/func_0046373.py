def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._level_default = self._mdata['level']['default_id_values'][0]
        self._start_time_default = self._mdata['start_time']['default_date_time_values'][0]
        self._grade_system_default = self._mdata['grade_system']['default_id_values'][0]
        self._items_shuffled_default = self._mdata['items_shuffled']['default_boolean_values'][0]
        self._score_system_default = self._mdata['score_system']['default_id_values'][0]
        self._deadline_default = self._mdata['deadline']['default_date_time_values'][0]
        self._duration_default = self._mdata['duration']['default_duration_values'][0]
        self._items_sequential_default = self._mdata['items_sequential']['default_boolean_values'][0]