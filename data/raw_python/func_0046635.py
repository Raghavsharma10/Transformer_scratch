def _init_metadata(self):
        """stub"""
        self._source_assessment_taken_id_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'source_assessment_taken_id'),
            'element_label': 'Source Assessment Taken ID that generated this one',
            'instructions': 'accepts a valid OSID Id string',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': []
        }