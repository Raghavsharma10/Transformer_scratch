def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._url_default = self._mdata['url']['default_string_values'][0]
        self._data_default = self._mdata['data']['default_object_values'][0]
        self._accessibility_type_default = self._mdata['accessibility_type']['default_type_values'][0]