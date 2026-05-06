def clear_choice(self, choice):
        """stub"""
        if len(self.my_osid_object_form._my_map['choices']) == 0:
            raise IllegalState('there are currently no choices defined for this question')
        if (len(self.my_osid_object_form._my_map['choices']) == 1 and
                choice in self.my_osid_object_form._my_map['choices']):
            raise IllegalState()
        self.my_osid_object_form._my_map['choices'] = \
            [c for c in self.my_osid_object_form._my_map['choices'] if c != choice]