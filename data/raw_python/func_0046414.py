def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['assessmentOfferedId'] = str(kwargs['assessment_offered_id'])
        self._my_map['takerId'] = self._taker_default
        self._my_map['assignedBankIds'] = [str(kwargs['bank_id'])]
        self._my_map['actualStartTime'] = None
        self._my_map['gradeId'] = ''
        self._my_map['completionTime'] = None
        self._my_map['score'] = 0.0