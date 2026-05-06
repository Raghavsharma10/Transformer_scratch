def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidContainableForm._init_metadata(self)
        osid_objects.OsidOperableForm._init_metadata(self)
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        if 'assessment_part_id' not in kwargs:
            # Only "Section" Parts are allowed directly under Assessments
            self._mdata['sequestered']['is_read_only'] = True
            self._mdata['sequestered']['is_required'] = True
            self._mdata['sequestered']['default_boolean_values'] = [False]
        else:
            if 'mdata' in kwargs:
                self._mdata['sequestered'] = kwargs['mdata']['sequestered']
        self._assessment_part_default = self._mdata['assessment_part']['default_id_values'][0]
        self._assessment_default = self._mdata['assessment']['default_id_values'][0]
        self._weight_default = self._mdata['weight']['default_cardinal_values'][0]
        self._allocated_time_default = self._mdata['allocated_time']['default_duration_values'][0]
        self._items_sequential_default = None
        self._items_shuffled_default = None
        self._mdata['children'] = {
            'element_label': 'Children',
            'instructions': 'accepts an IdList',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }