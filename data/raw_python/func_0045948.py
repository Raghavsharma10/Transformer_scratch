def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['url'] = self._url_default
        self._my_map['data'] = self._data_default
        self._my_map['accessibilityTypeId'] = self._accessibility_type_default
        self._my_map['assignedRepositoryIds'] = [str(kwargs['repository_id'])]
        self._my_map['assetId'] = str(kwargs['asset_id'])