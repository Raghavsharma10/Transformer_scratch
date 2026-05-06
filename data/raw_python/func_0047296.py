def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['nextAssessmentPartId'] = str(kwargs['next_assessment_part_id'])
        self._my_map['cumulative'] = self._cumulative_default
        self._my_map['minimumScore'] = self._minimum_score_default
        self._my_map['maximumScore'] = self._maximum_score_default
        self._my_map['assessmentPartId'] = str(kwargs['assessment_part_id'])
        self._my_map['assignedBankIds'] = [str(kwargs['bank_id'])]
        self._my_map['appliedAssessmentPartIds'] = []