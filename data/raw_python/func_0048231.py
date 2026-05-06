def clear_choice_ids(self):
        """stub"""
        if (self.get_choice_ids_metadata().is_read_only() or
                self.get_choice_ids_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['choiceIds'] = \
            self._choice_ids_metadata['default_object_values'][0]