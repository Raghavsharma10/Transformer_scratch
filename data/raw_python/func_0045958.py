def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""

        osid_objects.OsidContainableForm._init_map(self)
        osid_objects.OsidSourceableForm._init_map(self)
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['childIds'] = self._children_default
        self._my_map['assignedRepositoryIds'] = [str(kwargs['repository_id'])]