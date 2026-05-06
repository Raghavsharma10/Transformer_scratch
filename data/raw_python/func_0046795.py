def set_allow_repeat_items(self, allow_repeat_items):
        """determines if repeat items will be shown, or if the scaffold iteration will simply stop"""
        if self.get_allow_repeat_items_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_boolean(allow_repeat_items):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['allowRepeatItems'] = allow_repeat_items