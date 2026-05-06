def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._cumulative_default = self._mdata['cumulative']['default_boolean_values'][0]
        self._minimum_score_default = self._mdata['minimum_score']['default_cardinal_values'][0]
        self._maximum_score_default = self._mdata['maximum_score']['default_cardinal_values'][0]