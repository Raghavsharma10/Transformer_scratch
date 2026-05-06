def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['assignedBinIds'] = [str(kwargs['bin_id'])]
        self._my_map['group'] = self._group_default
        self._my_map['avatarId'] = self._avatar_default