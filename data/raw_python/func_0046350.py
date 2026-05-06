def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._learning_objectives_default = self._mdata['learning_objectives']['default_id_values']