def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidContainableForm._init_map(self)
        osid_objects.OsidOperableForm._init_map(self)
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        if 'assessment_part_id' in kwargs:
            self._my_map['assessmentPartId'] = str(kwargs['assessment_part_id'])
            if 'mdata' in kwargs:
                self._my_map['sequestered'] = kwargs['mdata']['sequestered']['default_boolean_values'][0]
        else:
            self._my_map['assessmentPartId'] = self._assessment_part_default
            self._my_map['sequestered'] = False  # Parts under Assessments must be "Sections"
        if 'assessment_id' in kwargs:
            self._my_map['assessmentId'] = str(kwargs['assessment_id'])
        else:
            self._my_map['assessmentId'] = self._assessment_default
        self._my_map['assignedBankIds'] = [str(kwargs['bank_id'])]
        self._my_map['allocatedTime'] = self._allocated_time_default
        self._my_map['itemsSequential'] = self._items_sequential_default
        self._my_map['itemsShuffled'] = self._items_shuffled_default
        self._my_map['weight'] = self._weight_default
        if self._supports_simple_sequencing():
            self._my_map['childIds'] = []