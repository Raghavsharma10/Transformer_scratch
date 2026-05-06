def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidRelationshipForm._init_metadata(self, **kwargs)
        self._completion_default = self._mdata['completion']['default_decimal_values'][0]
        self._level_default = self._mdata['level']['default_id_values'][0]