def _init_map(self):
        """stub"""
        super(SimpleDifficultyItemFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['texts']['difficulty'] = \
            self._difficulty_metadata['default_string_values'][0]