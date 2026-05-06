def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._courses_default = self._mdata['courses']['default_id_values']
        self._assessments_default = self._mdata['assessments']['default_id_values']
        self._assets_default = self._mdata['assets']['default_id_values']