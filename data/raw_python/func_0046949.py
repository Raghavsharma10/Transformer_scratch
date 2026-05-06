def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['assignedObjectiveBankIds'] = [str(kwargs['objective_bank_id'])]
        self._my_map['cognitiveProcessId'] = self._cognitive_process_default
        self._my_map['assessmentId'] = self._assessment_default
        self._my_map['knowledgeCategoryId'] = self._knowledge_category_default