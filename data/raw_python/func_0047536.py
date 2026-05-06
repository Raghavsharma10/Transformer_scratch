def _init_metadata(self):
        """stub"""
        super(EulerRotationAnswerFormRecord, self)._init_metadata()
        self._euler_rotation_metadata = {
            'element_id': Id(self.my_osid_object_form._authority,
                             self.my_osid_object_form._namespace,
                             'angle_values'),
            'element_label': 'Euler Angle Values',
            'instructions': 'Provide X, Y, and Z euler angle rotation values',
            'required': True,
            'read_only': False,
            'linked': True,
            'array': False,
            'default_object_values': [{}],
            'syntax': 'OBJECT',
            'object_set': []
        }