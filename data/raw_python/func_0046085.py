def _init_map(self):
        """stub"""
        self.my_osid_object_form._my_map['attempts'] = \
            int(self._attempts_metadata['default_object_values'][0])
        self.my_osid_object_form._my_map['weight'] = \
            float(self._weight_metadata['default_object_values'][0])
        # self.my_osid_object_form._my_map['rerandomize'] = \
        #     self._rerandomize_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['showanswer'] = \
            str(self._showanswer_metadata['default_object_values'][0])
        self.my_osid_object_form._my_map['markdown'] = \
            str(self._markdown_metadata['default_object_values'][0])