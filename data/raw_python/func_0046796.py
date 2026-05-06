def clear_allow_repeat_items(self):
        """reset allow repeat itmes to default value"""
        if (self.get_allow_repeat_items_metadata().is_read_only() or
                self.get_allow_repeat_items_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['allowRepeatItems'] = \
            bool(self._allow_repeat_items_metadata['default_boolean_values'][0])