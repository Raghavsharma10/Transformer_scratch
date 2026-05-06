def clear_solution(self):
        """stub"""
        if 'solution' not in self.my_osid_object_form._my_map:
            raise NotFound()
        self.my_osid_object_form._my_map['solution'] = \
            dict(self._solution_metadata['default_string_values'][0])