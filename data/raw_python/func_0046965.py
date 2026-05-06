def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['assignedObjectiveBankIds'] = [str(kwargs['objective_bank_id'])]
        self._my_map['courseIds'] = self._courses_default
        self._my_map['assessmentIds'] = self._assessments_default
        self._my_map['objectiveId'] = str(kwargs['objective_id'])
        self._my_map['assetIds'] = self._assets_default