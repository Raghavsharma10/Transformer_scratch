def _init_metadata(self):
        """stub"""
        super(BaseOrthoQuestionFormRecord, self)._init_metadata()
        self._ortho_view_set_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'ortho_view_set'),
            'element_label': 'Orthographic View Set',
            'instructions': '',
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_object_values': [''],
            'syntax': 'OBJECT',
            'object_set': []
        }