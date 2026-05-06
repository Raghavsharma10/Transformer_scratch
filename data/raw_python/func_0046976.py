def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidRelationshipForm._init_map(self, record_types=record_types)
        self._my_map['assignedObjectiveBankIds'] = [str(kwargs['objective_bank_id'])]
        self._my_map['completion'] = self._completion_default
        self._my_map['objectiveId'] = str(kwargs['objective_id'])
        self._my_map['resourceId'] = str(kwargs['resource_id'])
        self._my_map['levelId'] = self._level_default