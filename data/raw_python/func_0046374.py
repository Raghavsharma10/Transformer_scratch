def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['levelId'] = self._level_default
        self._my_map['startTime'] = self._start_time_default
        self._my_map['gradeSystemId'] = self._grade_system_default
        self._my_map['itemsShuffled'] = self._items_shuffled_default
        self._my_map['scoreSystemId'] = self._score_system_default
        self._my_map['deadline'] = self._deadline_default
        self._my_map['assignedBankIds'] = [str(kwargs['bank_id'])]
        self._my_map['duration'] = self._duration_default
        self._my_map['assessmentId'] = str(kwargs['assessment_id'])
        self._my_map['itemsSequential'] = self._items_sequential_default