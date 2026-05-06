def _init_metadata(self, **kwargs):
        """Initialize form metadata"""

        osid_objects.OsidContainableForm._init_metadata(self)
        osid_objects.OsidSourceableForm._init_metadata(self)
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._children_default = self._mdata['children']['default_id_values']