def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['rubricId'] = self._rubric_default
        self._my_map['assignedBankIds'] = [str(kwargs['bank_id'])]
        self._my_map['levelId'] = self._level_default
        if self._supports_simple_sequencing():
            self._my_map['childIds'] = []