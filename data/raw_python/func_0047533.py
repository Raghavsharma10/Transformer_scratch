def _init_metadata(self):
        """stub"""
        super(LabelOrthoFacesAnswerFormRecord, self)._init_metadata()
        self._face_values_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'face_values'),
            'element_label': 'Orthographic Face Values',
            'instructions': '',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }